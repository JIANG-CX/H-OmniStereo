# <div align = "center"><img src="figure/logo.png" alt="image-20200927095842317" width="8%" height="8%" /> H-OmniStereo: </div>

## <div align = "center">Zero-Shot Omnidirectional Stereo Matching with Heading-Aligned Normal Priors</div>

<div align="center">
<a href="https://arxiv.org/abs/2605.14963"><img src="https://img.shields.io/badge/ArXiv-2605.14963-da282a.svg"/></a>
</div>





> [Chenxing Jiang](https://jiang-cx.github.io/), Zhe Tong, Pusen Gao, Peize Liu, Yang Xu, Chuan Fang, Ping Tan, [Shaojie Shen †](https://uav.hkust.edu.hk/group/)

## Abstract

Stereo matching on top-bottom equirectangular images provides an effective framework for full-surround perception, as vertically aligned epipolar lines enable the use of advanced perspective stereo architectures that are largely driven by large-scale datasets and monocular priors. However, the performance of such adaptations is severely limited by the scarcity of omnidirectional stereo datasets and the degradation of perspective monocular priors under spherical distortions. To address these challenges, we propose H-OmniStereo, a zero-shot omnidirectional stereo matching framework. First, we construct high-quality synthetic dataset comprising over 2.8 million top-bottom equirectangular stereo pairs to scale up training. Second, we introduce an equirectangular monocular normal estimator, specifically operating in a heading-aligned coordinate system. Beyond providing distortion-robust and cross-view-consistent geometric priors for establishing reliable correspondences in stereo matching, this design boosts training efficiency and accommodates train-test FoV mismatches.  Extensive experiments show that our approach achieves higher accuracy than existing methods on out-of-domain datasets and successfully generalizes to real-world consumer camera setups using a single model. Both the model and the dataset will be open-sourced.

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
