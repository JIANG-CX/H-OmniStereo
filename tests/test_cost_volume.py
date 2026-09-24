import torch

from homnistereo.model.submodule import (
    build_concatenation_volume,
    build_groupwise_correlation_volume,
    groupwise_correlation,
)


def test_groupwise_volume_uses_fixed_left_reference_horizontal_shift():
    reference = torch.tensor(
        [[[[1.0, 2.0, 3.0]], [[4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0]], [[2.0, 3.0, 4.0]]]]
    )
    source = reference.flip(-1)

    volume = build_groupwise_correlation_volume(
        reference, source, max_disparity=3, groups=2
    )

    assert volume.shape == (1, 2, 3, 1, 3)
    assert torch.equal(
        volume[:, :, 0], groupwise_correlation(reference, source, groups=2)
    )
    assert torch.count_nonzero(volume[:, :, 1, :, :1]) == 0
    assert torch.equal(
        volume[:, :, 1, :, 1:],
        groupwise_correlation(reference[:, :, :, 1:], source[:, :, :, :-1], groups=2),
    )


def test_concatenation_volume_keeps_reference_and_shifts_source():
    reference = torch.tensor([[[[1.0, 2.0, 3.0]]]])
    source = torch.tensor([[[[4.0, 5.0, 6.0]]]])

    volume = build_concatenation_volume(reference, source, max_disparity=3)

    assert volume.shape == (1, 2, 3, 1, 3)
    assert torch.equal(volume[:, :1, 0], reference)
    assert torch.equal(volume[:, :1, 1], reference)
    assert torch.equal(volume[:, 1:, 0], source)
    assert torch.equal(volume[:, 1:, 1], torch.tensor([[[[0.0, 4.0, 5.0]]]]))
