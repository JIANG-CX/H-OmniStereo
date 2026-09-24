import torch

from homnistereo.model.normal_prior.spherical_decoder import SphericalFeatureDecoder


def test_prediction_head_returns_only_inference_outputs():
    decoder = SphericalFeatureDecoder(
        hidden_dim=16,
        num_heads=4,
        num_layers_head=(1, 1, 1, 1),
        out_dim=2,
    )
    decoder.set_original_shapes((8, 16))
    final_features = torch.randn(1, 2, 2, 4)

    normal, prior_features = decoder.prediction_head(final_features)

    assert normal.shape == (1, 3, 8, 16)
    assert prior_features.shape == final_features.shape

