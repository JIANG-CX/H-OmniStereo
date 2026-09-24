import math

import torch

from homnistereo.metrics import BatchWiseMetricAccumulator


def test_disparity_metrics_average_images_equally():
    predicted_disparity = torch.tensor([[[0.0, 2.0]], [[4.0, 4.0]]])
    target_disparity = torch.tensor([[[0.0, 0.0]], [[0.0, 0.0]]])
    predicted_depth = predicted_disparity.clone()
    target_depth = target_disparity.clone()
    valid_mask = torch.ones_like(predicted_disparity, dtype=torch.bool)

    accumulator = BatchWiseMetricAccumulator()
    accumulator.update(
        predicted_disparity,
        target_disparity,
        predicted_depth,
        target_depth,
        valid_mask,
    )
    result = accumulator.compute()

    assert result["disparity_mae"] == 2.5
    assert math.isclose(
        result["disparity_rmse"],
        (math.sqrt(2.0) + 4.0) / 2.0,
        abs_tol=1e-6,
    )
    assert result["bad_1px"] == 75.0
    assert result["bad_2px"] == 50.0
    assert result["bad_3px"] == 50.0
