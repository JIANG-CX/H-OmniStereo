# Inference examples

Each example directory contains an upper-camera image named `top`, a
lower-camera image named `bottom`, and a `camera.json` file. Image extensions
are preserved from the source data.

| Source | Example directories |
| --- | --- |
| 360SD-Net | `360sdnet/stairs`, `360sdnet/hall`, `360sdnet/room` |
| 3D60 | `threeD60/matterport3d_12`, `threeD60/stanford2d3d_area5a` |
| H-OmniStereo | `ours/val21_000094`, `ours/val73_000057` |
| MVS-GI | `mvsgi/cam1_2_pose_hard_001_000009`, `mvsgi/cam1_2_pose_easy_000_000006`, `mvsgi/cam2_0_pose_easy_003_000019` |

The camera file uses this schema:

```json
{
  "baseline": 0.2,
  "fov_params": [0.0, 360.0, 0.0, 180.0]
}
```

`baseline` is the vertical camera-center distance in meters. `fov_params`
contains `[lon_min, lon_max, lat_min, lat_max]` in degrees.
