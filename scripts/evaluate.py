#!/usr/bin/env python3
"""Evaluate H-OmniStereo with per-image (batch-wise) metric averaging."""

from __future__ import annotations

import argparse
import json
import os
import sys
from copy import deepcopy
from pathlib import Path

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from torch.utils.data import DataLoader, Subset  # noqa: E402
from tqdm import tqdm  # noqa: E402

from homnistereo.data.dataset import (  # noqa: E402
    EvaluationDataset,
    evaluation_collate,
    load_dataset_registry,
)
from homnistereo.metrics import BatchWiseMetricAccumulator  # noqa: E402
from homnistereo.predictor import HOmniStereoPredictor  # noqa: E402
from homnistereo.visualization import save_evaluation_visuals  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, default=PROJECT_ROOT / "checkpoints/h_omnistereo.pth"
    )
    parser.add_argument(
        "--dataset",
        default="threeD60",
        help="Exact dataset registry name (default: threeD60).",
    )
    parser.add_argument(
        "--dataset-config", type=Path, default=PROJECT_ROOT / "configs/datasets.json"
    )
    parser.add_argument("--root", type=Path, help="Override the selected dataset root.")
    parser.add_argument("--split", help="Override the selected dataset split file.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=22)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "outputs/evaluation"
    )
    parser.add_argument(
        "--visualize",
        type=int,
        default=10,
        metavar="N",
        help="Save the first N sample visualizations.",
    )
    parser.add_argument(
        "--max-samples", type=int, help="Limit samples for a smoke test."
    )
    parser.add_argument("--list-datasets", action="store_true")
    return parser.parse_args()


def _create_predictor(args: argparse.Namespace) -> HOmniStereoPredictor:
    return HOmniStereoPredictor(
        args.checkpoint,
        device=args.device,
        iterations=args.iterations,
    )


def main() -> None:
    args = parse_args()
    registry = load_dataset_registry(args.dataset_config)
    if args.list_datasets:
        print("\n".join(registry["datasets"]))
        return
    if args.dataset not in registry["datasets"]:
        supported = ", ".join(registry["datasets"])
        raise ValueError(
            f"Unknown dataset {args.dataset!r}. Supported datasets: {supported}"
        )
    dataset_name = args.dataset
    specification = deepcopy(registry["datasets"][dataset_name])
    if args.root is not None:
        specification["root"] = str(args.root)
    if args.split is not None:
        specification["split"] = args.split
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.workers < 0:
        raise ValueError("--workers cannot be negative.")
    if args.visualize < 0:
        raise ValueError("--visualize cannot be negative.")

    dataset = EvaluationDataset(dataset_name, specification)
    if args.max_samples is not None:
        if args.max_samples <= 0:
            raise ValueError("--max-samples must be positive.")
        dataset = Subset(dataset, range(min(args.max_samples, len(dataset))))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=args.device.startswith("cuda"),
        drop_last=False,
        collate_fn=evaluation_collate,
    )
    predictor = _create_predictor(args)
    metrics = BatchWiseMetricAccumulator()
    visualized = 0

    for batch in tqdm(loader, desc=f"Evaluating {dataset_name}"):
        prediction = predictor.predict(
            batch["top_image"],
            batch["bottom_image"],
            batch["baseline"],
            batch["fov_params"],
            depth_mask=batch["valid_mask"],
        )
        target_disparity = batch["disparity"].to(
            predictor.device, non_blocking=True
        )
        target_depth = batch["depth"].to(predictor.device, non_blocking=True)
        target_mask = batch["valid_mask"].to(predictor.device, non_blocking=True)
        metrics.update(
            prediction.disparity,
            target_disparity,
            prediction.depth,
            target_depth,
            target_mask,
        )

        for index in range(prediction.disparity.shape[0]):
            if visualized >= args.visualize:
                break
            sample_dir = args.output_dir / "visualizations" / f"{visualized:06d}"
            save_evaluation_visuals(
                sample_dir,
                {
                    "disparity": prediction.disparity[index].cpu().numpy(),
                    "depth": prediction.depth[index].cpu().numpy(),
                    "uncertainty": prediction.uncertainty[index].cpu().numpy(),
                },
                {
                    "disparity": target_disparity[index].cpu().numpy(),
                    "depth": target_depth[index].cpu().numpy(),
                },
                target_mask[index].cpu().numpy(),
            )
            (sample_dir / "sample.txt").write_text(
                batch["sample_id"][index] + "\n", encoding="utf-8"
            )
            visualized += 1

    results = {
        "architecture": "H-OmniStereo",
        "dataset": dataset_name,
        "dataset_root": specification["root"],
        "split": specification["split"],
        "samples": len(dataset),
        "checkpoint": str(args.checkpoint),
        "iterations": args.iterations,
        "batch_wise": metrics.compute(),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / f"{dataset_name}_batch_wise_metrics.json"
    result_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"Saved metrics to {result_path}")


if __name__ == "__main__":
    main()
