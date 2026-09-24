import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from .base import (
    fourier_dimension_expansion,
    flatten,
    DimensionAligner,
    AttentionSeq,
    ResidualUpsampler,
)
from homnistereo.geometry import amp_safe_normalize


class SphericalFeatureDecoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 512,
        num_heads: int = 8,
        expansion: int = 4,
        num_layers_head: tuple[int, int, int, int] = (2, 2, 2, 2),
        dropout: float = 0.0,
        kernel_size: int = 3,
        layer_scale: float = 0.0001,
        out_dim: int = 64,
        num_prompt_blocks: int = 1,
        use_norm: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

        self.up_sampler = nn.ModuleList([])
        self.pred_head = nn.ModuleList([])
        self.process_features = nn.ModuleList([])
        self.prompt_camera = nn.ModuleList([])
        mult = 2
        self.to_latents = nn.Linear(hidden_dim, hidden_dim)

        for _ in range(4):
            self.prompt_camera.append(
                AttentionSeq(
                    num_blocks=num_prompt_blocks,
                    dim=hidden_dim,
                    num_heads=num_heads,
                    expansion=expansion,
                    dropout=dropout,
                    layer_scale=-1.0,
                    context_dim=hidden_dim,
                )
            )

        for i, depth in enumerate(num_layers_head):
            current_dim = min(hidden_dim, mult * hidden_dim // int(2**i))
            next_dim = mult * hidden_dim // int(2 ** (i + 1))
            output_dim = max(next_dim, out_dim)
            self.process_features.append(
                nn.ConvTranspose2d(
                    hidden_dim,
                    current_dim,
                    kernel_size=2**i,
                    stride=2**i,
                    padding=0,
                )
            )
            self.up_sampler.append(
                ResidualUpsampler(
                    current_dim,
                    output_dim=output_dim,
                    expansion=expansion,
                    layer_scale=layer_scale,
                    kernel_size=kernel_size,
                    num_layers=depth,
                    use_norm=use_norm,
                )
            )
            pred_head = (
                nn.Sequential(nn.LayerNorm(next_dim), nn.Linear(next_dim, output_dim))
                if i == len(num_layers_head) - 1
                else nn.Identity()
            )
            self.pred_head.append(pred_head)

        self.to_normal_lr = nn.Conv2d(
            output_dim,
            output_dim // 2,
            kernel_size=3,
            padding=1,
            padding_mode="reflect",
        )
        self.to_normal_hr = nn.Sequential(
            nn.Conv2d(
                output_dim // 2, 32, kernel_size=3, padding=1, padding_mode="reflect"
            ),
            nn.LeakyReLU(),
            nn.Conv2d(32, 3, kernel_size=1),
        )

    def set_original_shapes(self, shapes: tuple[int, int]):
        self.original_shapes = shapes

    def set_shapes(self, shapes: tuple[int, int]):
        self.shapes = shapes

    def embed_sphere_dirs(self, sphere_dirs):
        sphere_embedding = flatten(
            sphere_dirs, old=self.original_shapes, new=self.shapes
        )
        # index 0 -> Y
        # index 1 -> Z
        # index 2 -> X
        r1, r2, r3 = (
            sphere_embedding[..., 0],
            sphere_embedding[..., 1],
            sphere_embedding[..., 2],
        )
        polar = torch.asin(r2)
        r3_clipped = r3.abs().clip(min=1e-5) * (2 * (r3 >= 0).int() - 1)
        azimuth = torch.atan2(r1, r3_clipped)
        # [polar, azimuth] is the angle field
        sphere_embedding = torch.stack([polar, azimuth], dim=-1)
        # expand the dimension of the angle field to image feature dimensions, via sine-cosine basis embedding
        sphere_embedding = fourier_dimension_expansion(
            sphere_embedding,
            dim=self.hidden_dim,
            max_freq=max(self.shapes) // 2,
            use_cos=False,
        )
        return sphere_embedding

    def condition(self, feat, sphere_embeddings):
        conditioned_features = [
            prompter(rearrange(feature, "b h w c -> b (h w) c"), sphere_embeddings)
            for prompter, feature in zip(self.prompt_camera, feat)
        ]
        return conditioned_features

    def process(self, features_list, sphere_embeddings):
        conditioned_features = self.condition(features_list, sphere_embeddings)
        init_latents = self.to_latents(conditioned_features[0])
        init_latents = rearrange(
            init_latents, "b (h w) c -> b c h w", h=self.shapes[0], w=self.shapes[1]
        ).contiguous()
        conditioned_features = [
            rearrange(
                x, "b (h w) c -> b c h w", h=self.shapes[0], w=self.shapes[1]
            ).contiguous()
            for x in conditioned_features
        ]
        latents = init_latents

        # Pyramid-like multi-layer convolutional feature extraction
        for i, up in enumerate(self.up_sampler):
            # Select skip connection feature
            # Reuse the last available feature if we go beyond the list
            idx = min(i + 1, len(conditioned_features) - 1)
            latents = latents + self.process_features[i](conditioned_features[idx])

            latents = up(latents)
        return latents

    def prediction_head(
        self, final_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        height, width = final_features.shape[-2:]
        projected_features = self.pred_head[-1](
            final_features.permute(0, 2, 3, 1)
        ).permute(0, 3, 1, 2)
        normal_features = F.interpolate(
            projected_features,
            size=(height, width),
            mode="bilinear",
            align_corners=True,
        )
        prior_features = F.interpolate(
            projected_features,
            size=(height, width),
            mode="bilinear",
            align_corners=True,
        ).clone()
        normal = self.to_normal_lr(normal_features)
        normal = F.interpolate(
            normal, size=self.original_shapes, mode="bilinear", align_corners=True
        )
        normal = self.to_normal_hr(normal)
        normal = amp_safe_normalize(normal, dim=1, eps=1e-4)
        return normal, prior_features

    def forward(
        self,
        features: list[torch.Tensor],
        sphere_dirs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sphere_embeddings = self.embed_sphere_dirs(sphere_dirs)
        final_features = self.process(features, sphere_embeddings)
        return self.prediction_head(final_features)


class SphericalNormalDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.dim_aligner = DimensionAligner(
            input_dims=[1024, 1024, 1024, 1024],
            hidden_dim=512,
        )

        self.feature_decoder = SphericalFeatureDecoder()

    def forward(
        self, images, features, sphere_dirs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _, _, H, W = images.shape
        common_shape = features[0].shape[1:3]
        features = self.dim_aligner(features)

        sphere_dirs = rearrange(sphere_dirs, "b c h w -> b (h w) c")

        self.feature_decoder.set_shapes(common_shape)
        self.feature_decoder.set_original_shapes((H, W))
        return self.feature_decoder(
            features=features,
            sphere_dirs=sphere_dirs,
        )
