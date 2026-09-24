import numpy as np

from homnistereo.data.dataset import _prepare_sample
from homnistereo.data.types import RawSample


def _raw_erp_sample() -> RawSample:
    height, width = 32, 64
    return RawSample(
        top_image=np.zeros((height, width, 3), dtype=np.uint8),
        bottom_image=np.zeros((height, width, 3), dtype=np.uint8),
        disparity=np.ones((height, width), dtype=np.float32),
        valid_mask=np.ones((height, width), dtype=bool),
        depth=np.ones((height, width), dtype=np.float32),
        baseline=0.2,
        fov_params=(0.0, 360.0, 0.0, 180.0),
    )


def _specification() -> dict:
    return {
        "image_size": [64, 32],
        "force_2_to_1": True,
    }


def test_stereo_mask_excludes_erp_poles():
    sample = _prepare_sample(
        _raw_erp_sample(), _specification(), "sample"
    )

    assert not sample["valid_mask"][0].any()
    assert not sample["valid_mask"][-1].any()
    assert sample["valid_mask"][1:-1].all()
