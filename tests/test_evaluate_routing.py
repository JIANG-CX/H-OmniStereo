from argparse import Namespace
from pathlib import Path

from scripts import evaluate


def _arguments() -> Namespace:
    return Namespace(
        checkpoint=Path("stereo.pth"),
        device="cuda",
        iterations=22,
    )


def test_evaluation_uses_stereo_predictor(monkeypatch):
    calls = []

    def make_stereo(checkpoint, *, device, iterations):
        calls.append((checkpoint, device, iterations))
        return object()

    monkeypatch.setattr(evaluate, "HOmniStereoPredictor", make_stereo)
    predictor = evaluate._create_predictor(_arguments())

    assert predictor is not None
    assert calls == [(Path("stereo.pth"), "cuda", 22)]
