"""ERP geometry shared by public inference and dataset evaluation."""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def amp_safe_normalize(inputs, dim=-1, eps=1e-4):
    original_dtype = inputs.dtype
    with torch.amp.autocast("cuda", enabled=False):
        inputs = torch.nan_to_num(inputs.float(), nan=1, neginf=1, posinf=1)
        output = F.normalize(inputs, dim=dim, eps=eps)
        output = torch.nan_to_num(output, nan=1, neginf=1, posinf=1)
    return output.to(original_dtype)


def normalize_image(image: torch.Tensor) -> torch.Tensor:
    """Normalize RGB tensors whose values are in [0, 255]."""
    mean = image.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
    std = image.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
    return ((image / 255.0 - mean) / std).contiguous()


def get_resize_keep_aspect_ratio(height, width, divider=16, max_H=1232, max_W=1232):
    if max_H % divider or max_W % divider:
        raise ValueError("Maximum dimensions must be divisible by divider.")

    def rounded(value):
        return int(np.ceil(value / divider) * divider)

    resized_height, resized_width = rounded(height), rounded(width)
    if resized_height > max_H or resized_width > max_W:
        if resized_height > resized_width:
            resized_width = rounded(resized_width * max_H / resized_height)
            resized_height = max_H
        else:
            resized_height = rounded(resized_height * max_W / resized_width)
            resized_width = max_W
    return resized_height, resized_width


def _pixel_grid(batch_size, height, width, device, dtype):
    x = torch.linspace(0.5, width - 0.5, width, device=device)
    y = torch.linspace(0.5, height - 0.5, height, device=device)
    grid_x = x.repeat(height, 1)
    grid_y = y.repeat(width, 1).t()
    return (
        torch.stack([grid_x, grid_y], dim=0)
        .float()
        .unsqueeze(0)
        .repeat(batch_size, 1, 1, 1)
        .to(device=device, dtype=dtype)
    )


def _erp_rays(batch_size, height, width, fov_params, only_lat):
    grid = _pixel_grid(batch_size, height, width, fov_params.device, fov_params.dtype)
    x_pixel, y_pixel = grid.unbind(dim=1)
    lon_min, lon_max, lat_min, lat_max = fov_params.unbind(dim=1)
    longitude = (
        lon_min[:, None, None]
        - 180.0
        + x_pixel / (width - 1.0) * (lon_max - lon_min)[:, None, None]
    )
    latitude = (
        lat_min[:, None, None]
        - 90.0
        + y_pixel / (height - 1.0) * (lat_max - lat_min)[:, None, None]
    )
    longitude = longitude * np.pi / 180.0
    latitude = latitude * np.pi / 180.0
    if only_lat:
        longitude *= 0.0
    rays = torch.stack(
        [
            torch.cos(latitude) * torch.sin(longitude),
            torch.sin(latitude),
            torch.cos(latitude) * torch.cos(longitude),
        ],
        dim=1,
    )
    return rays / torch.norm(rays, dim=1, keepdim=True).clip(min=1e-5)


def get_rays(batch_size, height, width, fov_params, only_lat=False):
    """Return unit camera rays for ERP samples."""
    return _erp_rays(batch_size, height, width, fov_params, only_lat)


def _optimal_vertical_padding(height, lat_top, lat_bottom, padding):
    if padding <= 0:
        return 0, 0
    if height == 1 or np.isclose(lat_top, lat_bottom):
        top = padding // 2
        return top, padding - top
    step = (lat_bottom - lat_top) / (height - 1)
    new_center = (height + padding - 1) / 2.0
    ideal_top = (lat_top + new_center * step - 90.0) / step
    candidates = {0, padding, int(np.floor(ideal_top)), int(np.ceil(ideal_top))}
    candidates = [min(padding, max(0, value)) for value in candidates]
    top = min(
        candidates,
        key=lambda value: (
            abs(lat_top - value * step + new_center * step - 90.0),
            abs(value - padding / 2.0),
        ),
    )
    return top, padding - top


def compute_padding_params(
    original_h,
    original_w,
    target_width,
    target_height,
    lat_degree_range=None,
    lon_degree_range=None,
):
    """Compute ERP padding while keeping the requested FoV centered."""
    current_ratio = original_w / original_h
    target_ratio = target_width / target_height
    if abs(current_ratio - target_ratio) <= 1e-6:
        top = bottom = left = right = 0
        padded_h, padded_w = original_h, original_w
    elif current_ratio < target_ratio:
        padded_w = int(original_h * target_ratio)
        left = (padded_w - original_w) // 2
        right = padded_w - original_w - left
        top = bottom = 0
        padded_h = original_h
    else:
        padded_h = int(original_w / target_ratio)
        padding = padded_h - original_h
        if lat_degree_range is not None:
            top, bottom = _optimal_vertical_padding(
                original_h, lat_degree_range[0], lat_degree_range[1], padding
            )
        else:
            top, bottom = padding // 2, padding - padding // 2
        left = right = 0
        padded_w = original_w
    result = {
        "pad_top": top,
        "pad_bottom": bottom,
        "pad_left": left,
        "pad_right": right,
        "padded_h": padded_h,
        "padded_w": padded_w,
        "need_padding": any((top, bottom, left, right)),
    }
    if lat_degree_range is not None and lon_degree_range is not None:
        lat_scope = lat_degree_range[1] - lat_degree_range[0]
        lon_scope = lon_degree_range[1] - lon_degree_range[0]
        result["new_lat_degree_range"] = [
            lat_degree_range[0] - top / (original_h - 1) * lat_scope,
            lat_degree_range[1] + bottom / (original_h - 1) * lat_scope,
        ]
        result["new_lon_degree_range"] = [
            lon_degree_range[0] - left / (original_w - 1) * lon_scope,
            lon_degree_range[1] + right / (original_w - 1) * lon_scope,
        ]
    return result


def padding_resize_image(
    image,
    target_width,
    target_height,
    padding_mode="edge",
    pad_params=None,
    original_h=None,
    original_w=None,
    interpolation=cv2.INTER_LINEAR,
):
    if image is None:
        return None
    height, width = image.shape[:2]
    if pad_params is None:
        pad_params = compute_padding_params(
            original_h or height,
            original_w or width,
            target_width,
            target_height,
        )
    if pad_params["need_padding"]:
        spatial_padding = (
            (pad_params["pad_top"], pad_params["pad_bottom"]),
            (pad_params["pad_left"], pad_params["pad_right"]),
        )
        padding = spatial_padding if image.ndim == 2 else (*spatial_padding, (0, 0))
        mode = "edge" if padding_mode == "edge" else "constant"
        image = np.pad(image, padding, mode=mode)
    resized = cv2.resize(
        image, (target_width, target_height), interpolation=interpolation
    )
    if image.ndim == 3 and resized.ndim == 2:
        resized = resized[..., None]
    return resized


def depth_uint8_decoding(encoded_depth, scale=1000):
    encoded_depth = encoded_depth.astype(float)
    return (
        encoded_depth[..., 0] * 255 * 255
        + encoded_depth[..., 1] * 255
        + encoded_depth[..., 2]
    ) / float(scale)


def depth_to_disp_pix(depth, baseline, fov_params):
    latitude = fov_params[2:]
    height, width = depth.shape
    y = (
        np.linspace(height - 0.5, 0.5, num=height)
        .reshape(height, 1)
        .repeat(width, axis=1)
    )
    theta = y * ((latitude[1] - latitude[0]) * (np.pi / 180) / height)
    theta += (180 - latitude[1]) * (np.pi / 180)
    disparity = np.arctan2(np.sin(theta), depth / baseline + np.cos(theta))
    disparity = np.where(disparity < 0, disparity + np.pi, disparity)
    return disparity * height / np.pi


def disp_to_depth(disparity, baseline, valid_mask, fov_params):
    """Convert bottom-referenced ERP disparity to radial depth."""
    if isinstance(disparity, np.ndarray):
        output = disp_to_depth(
            torch.from_numpy(disparity).unsqueeze(0),
            torch.as_tensor(baseline).reshape(1),
            torch.from_numpy(valid_mask).unsqueeze(0),
            torch.as_tensor(fov_params).unsqueeze(0),
        )
        return output[0].numpy()

    _, height, width = disparity.shape
    output = torch.zeros_like(disparity)
    for index in range(disparity.shape[0]):
        valid = valid_mask[index] & (disparity[index] > 1e-4)
        if not torch.any(valid):
            continue
        lat_min, lat_max = fov_params[index, 2:]
        y = (
            torch.linspace(
                height - 0.5,
                0.5,
                steps=height,
                device=disparity.device,
                dtype=disparity.dtype,
            )
            .view(height, 1)
            .expand(height, width)
        )
        lat_range = (lat_max - lat_min) * (torch.pi / 180.0)
        theta = y * (lat_range / height) + (180.0 - lat_max) * (torch.pi / 180.0)
        disparity_radians = disparity[index] * (lat_range / height)
        valid &= disparity_radians > 1e-4
        output[index, valid] = (
            torch.sin(theta[valid])
            / torch.tan(disparity_radians[valid].clamp_min(1e-4))
            - torch.cos(theta[valid])
        ) * baseline[index]
    return output.clamp_(0, 1e4)


def depth_to_points(depth, valid_mask=None, fov_params=None):
    """Convert ERP radial depth to dense XYZ maps or valid NumPy XYZ rows."""
    numpy_input = isinstance(depth, np.ndarray)
    if numpy_input:
        depth = torch.from_numpy(depth).unsqueeze(0)
        mask = None if valid_mask is None else torch.from_numpy(valid_mask).unsqueeze(0)
        params = torch.as_tensor(fov_params, dtype=depth.dtype).unsqueeze(0)
    else:
        mask, params = valid_mask, fov_params
    batch_size, height, width = depth.shape
    grid = _pixel_grid(batch_size, height, width, depth.device, depth.dtype)
    output = torch.zeros(
        (batch_size, height, width, 3), device=depth.device, dtype=depth.dtype
    )
    for index in range(batch_size):
        x, y = grid[index]
        lon_min, lon_max, lat_min, lat_max = params[index]
        longitude = torch.deg2rad(lon_min + x / width * (lon_max - lon_min) - 180)
        latitude = torch.deg2rad(lat_min + y / height * (lat_max - lat_min) - 90)
        direction = torch.stack(
            [
                torch.cos(latitude) * torch.sin(longitude),
                torch.sin(latitude),
                torch.cos(latitude) * torch.cos(longitude),
            ],
            dim=-1,
        )
        output[index] = direction * depth[index, ..., None]
    if mask is not None:
        output[~mask] = 0
    if numpy_input:
        points = output[0].numpy()
        return points[valid_mask] if valid_mask is not None else points.reshape(-1, 3)
    return output
