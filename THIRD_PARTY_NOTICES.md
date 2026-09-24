# Third-party notices

H-OmniStereo includes adapted inference components from the following projects:

- FoundationStereo, distributed under the research/evaluation-only license in
  [LICENSE](LICENSE). The released stereo architecture is a fixed, inference-only
  derivative of that implementation.
- The heading-aligned normal prior implementation in `homnistereo/model/normal_prior`
  is adapted from the project's Apache-2.0 normal-estimation code. Its license is
  reproduced in `licenses/HEADING_NORMAL_PRIOR_LICENSE`.
- The vendored DINOv2 transformer implementation under
  `homnistereo/model/normal_prior/dinov2` is distributed under Apache-2.0. Its
  license is reproduced in `licenses/DINOV2_LICENSE`.
- EdgeNeXt is instantiated through the external `timm` dependency; its source is
  not vendored in this repository.
- The public dataset preparation scripts follow the data formats and geometric
  conventions released by [3D60](https://github.com/VCL3D/3D60) and
  [MVS-GI](https://github.com/castacks/mvs_gi).

These notices do not replace the terms in the corresponding license files.
