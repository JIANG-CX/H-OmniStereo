# <div align = "center"><img src="figure/logo.png" alt="image-20200927095842317" width="5%" height="5%" /> H-OmniStereo: </div>

### <div align = "center">Zero-Shot Omnidirectional Stereo Matching with Heading-Aligned Normal Priors</div>

<div align="center">
<a href="https://arxiv.org/abs/2605.14963"><img src="https://img.shields.io/badge/ArXiv-2605.14963-da282a.svg"/></a>
<a href="https://youtu.be/prwpXCU00oE"><img alt="Youtube" src="https://img.shields.io/badge/Video-Youtube-red"/></a>
</div>

> [Chenxing Jiang](https://jiang-cx.github.io/), Zhe Tong, Pusen Gao, Peize Liu, Yang Xu, Chuan Fang, Ping Tan, [Shaojie Shen †](https://uav.hkust.edu.hk/group/)

## Abstract

Stereo matching on top-bottom equirectangular images provides an effective framework for full-surround perception, as vertically aligned epipolar lines enable the use of advanced perspective stereo architectures that are largely driven by large-scale datasets and monocular priors. However, the performance of such adaptations is severely limited by the scarcity of omnidirectional stereo datasets and the degradation of perspective monocular priors under spherical distortions. To address these challenges, we propose H-OmniStereo, a zero-shot omnidirectional stereo matching framework. First, we construct high-quality synthetic dataset comprising over 2.8 million top-bottom equirectangular stereo pairs to scale up training. Second, we introduce an equirectangular monocular normal estimator, specifically operating in a heading-aligned coordinate system. Beyond providing distortion-robust and cross-view-consistent geometric priors for establishing reliable correspondences in stereo matching, this design boosts training efficiency and accommodates train-test FoV mismatches.  Extensive experiments show that our approach achieves higher accuracy than existing methods on out-of-domain datasets and successfully generalizes to real-world consumer camera setups using a single model. Both the model and the dataset will be open-sourced.

https://github.com/user-attachments/assets/6cc2f22a-f19d-433c-80fe-949178a36a1d

## Environment setup

Requirements:

- Linux
- Conda
- NVIDIA GPU and a driver compatible with CUDA 12.1

The PyTorch wheel includes the CUDA runtime; a system CUDA toolkit is not
required.

Create the environment from the repository root:

```bash
cd /path/to/H-OmniStereo
conda env create -f environment.yml
conda activate h-omnistereo
git lfs install
git lfs pull
python -m pip check
python scripts/evaluate.py --list-datasets
```

Confirm that PyTorch can access the GPU:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.get_device_name())"
```

The model checkpoint is stored at `checkpoints/h_omnistereo.pth`.

## H-OmniStereo Dataset

The H-OmniStereo dataset contains 2.8M vertically displaced ERP
stereo pairs. The published training split is available at
`splits/ours/train.txt`.

### Download

Download the dataset from
[Hugging Face](https://huggingface.co/datasets/JCX1999/H-OmniStereo/tree/main):

```bash
python -m pip install -U huggingface_hub
hf download JCX1999/H-OmniStereo \
  --repo-type dataset \
  --local-dir H-OmniStereo-Dataset
```

### Directory structure

After extracting the dataset, organize the scene directories and training
split under a common root:

```text
H-OmniStereo-Dataset/
├── train.txt
└── <scene>/
    └── texture_set_<id>/
        ├── <frame>_up.jpg
        ├── <frame>_down.jpg
        ├── <frame>_updisp.png
        ├── <frame>_downdisp.png
        ├── <frame>_upnormal.jpg
        ├── <frame>_downnormal.jpg
        └── <frame>_camera_params.json
```

Each line in `train.txt` identifies one sample without a file suffix:

```text
<scene>/texture_set_<id>/<frame>
```

The `up` and `down` files contain the ERP stereo pair, `updisp` and `downdisp`
contain the encoded disparity maps, and `upnormal` and `downnormal` contain the
surface-normal maps. The camera JSON records the stereo baseline, resolution,
and encoding metadata for the sample.

### Reading disparity maps

```python
import imageio.v2 as imageio
import numpy as np

# RGB base-255 encoding; the decoded subpixel unit is 1/4096 pixel.
encoded_rgb = imageio.imread("<frame>_downdisp.png")[..., :3].astype(np.float64)
vertical_disparity_px = (
    encoded_rgb[..., 0] * 255.0 * 255.0
    + encoded_rgb[..., 1] * 255.0
    + encoded_rgb[..., 2]
) / 4096.0

height = vertical_disparity_px.shape[0]
valid_mask = (
    (vertical_disparity_px > 1e-7)
    & (vertical_disparity_px < height / 2.0)
)

# Latitude-direction correspondence displacement for the down reference view.
# A full ERP spans 180 degrees vertically; this value is not metric depth.
vertical_disparity_degrees = vertical_disparity_px * (180.0 / height)
```

Use `<frame>_updisp.png` in the same way when `up.jpg` is the reference view.
Repository references: [`depth_uint8_decoding`](homnistereo/geometry.py#L204-L210),
the [`H-OmniStereo dataset loader`](homnistereo/data/loaders/generated_omnidirectional.py#L17-L51),
and [`disp_to_depth`](homnistereo/geometry.py#L228-L266) for spherical radial
depth conversion using the camera baseline.

### Reading normal maps

```python
from PIL import Image
import numpy as np

# Use <frame>_upnormal.jpg for the up reference view.
encoded_normal = np.asarray(
    Image.open("<frame>_downnormal.jpg").convert("RGB"),
    dtype=np.float32,
)

# Channel axes: x points right, y points forward, and z points up.
normal_xyz = encoded_normal / 255.0 * 2.0 - 1.0

# Combine normal validity with valid_mask from the disparity map above.
normal_valid_mask = (
    valid_mask
    & ~np.isnan(normal_xyz).any(axis=2)
    & (np.abs(normal_xyz) > (128.0 / 255.0 * 2.0 - 1.0)).any(axis=2)
)
normal_xyz = np.nan_to_num(normal_xyz, nan=1.0, posinf=1.0, neginf=1.0)
```

## Two-image inference

Inference accepts any same-size top/bottom equirectangular stereo pair. The two
images must be RGB panoramas from a vertically displaced camera pair and must
cover the same field of view. Image dimensions are adjusted internally to the
next compatible network dimensions when they are not divisible by 32.

Each image pair is accompanied by a JSON file containing its camera geometry.
The bundled H-OmniStereo example uses:

```python
top_image_path = "assets/ours/val73_000057/top.jpg"
bottom_image_path = "assets/ours/val73_000057/bottom.jpg"
depth_path = None
camera_params_path = "assets/ours/val73_000057/camera.json"
```

- `top_image_path`: path to the upper-camera ERP image.
- `bottom_image_path`: path to the lower-camera ERP image used as the output
  reference view.
- `depth_path`: optional ground-truth depth path for external comparison.
  H-OmniStereo inference does not require it, so it is `None` in this example.
- `camera_params_path`: path to a JSON file containing `baseline`, the vertical
  camera-center distance in meters, and `fov_params`, the ERP bounds in degrees
  ordered as `[lon_min, lon_max, lat_min, lat_max]`.

The example camera file is:

```json
{
  "baseline": 0.41977808801565253,
  "fov_params": [0.0, 360.0, 0.0, 180.0]
}
```

Run the example with:

```bash
python scripts/infer.py \
  --top assets/ours/val73_000057/top.jpg \
  --bottom assets/ours/val73_000057/bottom.jpg \
  --camera-params assets/ours/val73_000057/camera.json \
  --output-dir outputs/inference/val73_000057
```

Additional pairs are organized by source under `assets`:

| Source       | Example directories                                          |
| ------------ | ------------------------------------------------------------ |
| 360SD-Net    | `assets/360sdnet/stairs`, `assets/360sdnet/hall`, `assets/360sdnet/room` |
| 3D60         | `assets/threeD60/matterport3d_12`, `assets/threeD60/stanford2d3d_area5a` |
| H-OmniStereo | `assets/ours/val21_000094`, `assets/ours/val73_000057`       |
| MVS-GI       | `assets/mvsgi/cam1_2_pose_hard_001_000009`, `assets/mvsgi/cam1_2_pose_easy_000_000006`, `assets/mvsgi/cam2_0_pose_easy_003_000019` |

Each directory has the same top/bottom/camera-file layout. The complete asset
index is also available in [`assets/README.md`](assets/README.md).

Inference writes NumPy arrays and color visualizations for disparity, depth,
and uncertainty. It also writes a PLY point cloud by default; use
`--no-point-cloud` to disable it.

## Dataset evaluation

The default dataset is `threeD60`. The supported dataset names are `threeD60`,
`threeD60warp`, `mvsgi`, `ours`, and `deep360`. Names are case-sensitive. List
the available datasets with:

```bash
python scripts/evaluate.py --list-datasets
```

### Dataset preparation

The paper test splits are included in the repository:

| Dataset   | Split path                           | Samples |
| --------- | ------------------------------------ | ------: |
| 3D60      | `splits/threeD60/test_index.txt`     |   2,189 |
| 3D60-Warp | `splits/threeD60warp/test_index.txt` |   2,189 |
| MVS-GI    | `splits/mvsgi/test_index.txt`        |  26,339 |

#### 3D60 and 3D60-Warp

Download the original spherical panoramas using the instructions from the
[3D60 repository](https://github.com/VCL3D/3D60). Preserve the view and source
directories so the relevant part of the extracted dataset has this layout:

```text
3D60/
├── Center_Left_Down/
│   └── <source>/
│       ├── <sample>_color_0_Left_Down_0.0.png
│       └── <sample>_depth_0_Left_Down_0.0.exr
├── Right/
│   └── <source>/
│       ├── <sample>_color_0_Right_0.0.png
│       └── <sample>_depth_0_Right_0.0.exr
└── Up/
    └── <source>/
        ├── <sample>_color_0_Up_0.0.png
        └── <sample>_depth_0_Up_0.0.exr
```

`<source>` is one of the downloaded source directories such as
`Matterport3D`, `Stanford2D3D`, or `SunCG`. The original 3D60 evaluator reads
`Center_Left_Down` and `Up` directly.

Create the rotated top/bottom 3D60-Warp test set with:

```bash
python scripts/preprocess_3d60_warp.py \
  --input-root /path/to/3D60 \
  --output-root /path/to/new/3D60_Warp
```

The script applies the paper's 45-degree ERP rotation to the `Right` and `Up`
views and copies the released split into the new output root:

```text
3D60_Warp/
├── test_index.txt
├── Right_warp/
│   └── <source>/
│       ├── <sample>_color_0_Right_warp_0.0.png
│       └── <sample>_depth_0_Right_warp_0.0.exr
└── Up_warp/
    └── <source>/
        ├── <sample>_color_0_Up_warp_0.0.png
        └── <sample>_depth_0_Up_warp_0.0.exr
```

CUDA is used by default. Add `--device cpu` when CUDA preprocessing is not
available.

#### MVS-GI

Download and extract the fisheye data following the
[MVS-GI repository](https://github.com/castacks/mvs_gi). Keep the duplicated
scene directory level and the original `cam0`, `cam1`, and `cam2` names:

```text
MVS_GI/
└── <scene>_dsta_data/
    └── <scene>_dsta_data/
        └── <pose>/
            ├── cam0/
            │   ├── mask.png
            │   ├── <frame>_Fisheye.png
            │   └── <frame>_FisheyeDistance.png
            ├── cam1/
            │   ├── <frame>_Fisheye.png
            │   └── <frame>_FisheyeDistance.png
            └── cam2/
                ├── <frame>_Fisheye.png
                └── <frame>_FisheyeDistance.png
```

Convert the released test split from 195-degree equidistant fisheye images to
the central 180-degree, 512 x 512 ERP representation with:

```bash
python scripts/preprocess_mvsgi.py \
  --input-root /path/to/MVS_GI \
  --output-root /path/to/new/MVS_GI_Warp_512
```

The output contains all three camera pairs used by the evaluator and a copy of
`splits/mvsgi/test_index.txt`:

```text
MVS_GI_Warp_512/
├── test_index.txt
├── UP/
│   ├── CAM1_0/<scene>/<pose>/<frame>_up_{rgb.jpg,depth.png}
│   ├── CAM2_0/<scene>/<pose>/<frame>_up_{rgb.jpg,depth.png}
│   └── CAM1_2/<scene>/<pose>/<frame>_up_{rgb.jpg,depth.png}
└── DOWN/
    ├── CAM1_0/<scene>/<pose>/<frame>_down_{rgb.jpg,depth.png}
    ├── CAM2_0/<scene>/<pose>/<frame>_down_{rgb.jpg,depth.png}
    └── CAM1_2/<scene>/<pose>/<frame>_down_{rgb.jpg,depth.png}
```

Depth maps are stored as packed float32 PNG files.

### Running evaluation

Evaluate 3D60 with:

```bash
python scripts/evaluate.py \
  --dataset threeD60 \
  --root /path/to/3D60 \
  --split /path/to/H-OmniStereo/splits/threeD60/test_index.txt
```

Evaluate the generated 3D60-Warp dataset with:

```bash
python scripts/evaluate.py \
  --dataset threeD60warp \
  --root /path/to/3D60_Warp \
  --split /path/to/H-OmniStereo/splits/threeD60warp/test_index.txt
```

Evaluate the generated MVS-GI dataset with:

```bash
python scripts/evaluate.py \
  --dataset mvsgi \
  --root /path/to/MVS_GI_Warp_512 \
  --split /path/to/H-OmniStereo/splits/mvsgi/test_index.txt
```

Edit the placeholder roots in `configs/datasets.json` or pass `--root` and
`--split` on the command line. Evaluation outputs contain only per-image
equal-weight batch-wise disparity/depth metrics and optional sample
visualizations. Use `--visualize 0` to disable visualizations.

## Acknowledgements

We adapt code, model components, and data conventions from several remarkable
open-source projects, including
[FoundationStereo](https://github.com/NVlabs/FoundationStereo/),
[DA-2](https://github.com/EnVision-Research/DA-2),
[3D60](https://github.com/VCL3D/3D60), and
[MVS-GI](https://github.com/castacks/mvs_gi). We thank their authors for making
their work publicly available.

## Contact

You can contact the author through email: cjiangan@connect.ust.hk

## Citing

If you find our work useful, please consider citing:

```
@misc{jiang2026homnistereozeroshotomnidirectionalstereo,
      title={H-OmniStereo: Zero-Shot Omnidirectional Stereo Matching with Heading-Aligned Normal Priors}, 
      author={Chenxing Jiang and Zhe Tong and Pusen Gao and Peize Liu and Yang Xu and Chuan Fang and Ping Tan and Shaojie Shen},
      year={2026},
      eprint={2605.14963},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2605.14963}, 
}
```
