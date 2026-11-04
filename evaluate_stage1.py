"""Run the locked-input PeMS08 development baseline comparison."""
import json
from pathlib import Path

import numpy as np

from experiment import ridge, score

OUTPUT = Path("results/staged")


def paired_summary(prediction, reference, actual, mask, blocks, draws):
    difference = np.abs(prediction - actual) - np.abs(reference - actual)
    numerator = np.bincount(blocks, weights=(difference * mask).sum(axis=1))
    denominator = np.bincount(blocks, weights=mask.sum(axis=1))
    counts = denominator[draws].sum(axis=1)
    bootstrap = numerator[draws].sum(axis=1)[counts > 0] / counts[counts > 0]
    signed = (prediction - actual)[mask]
    return {
        "mae_difference": float(difference[mask].mean()),
        "day_block_95_percent_interval": np.quantile(bootstrap, [.025, .975]).tolist(),
        "underprediction_rate": float((signed < 0).mean()),
        "mean_underprediction_when_low": float(-signed[signed < 0].mean()) if (signed < 0).any() else None,
        "n": int(mask.sum()),
    }


def load_split(dataset="data/PEMS08.npz", external=False):
    with np.load(dataset) as data:
        values = data["data"]
        flow = (values[:, :, 0] if values.ndim == 3 else values).astype(np.float64)
    first, second = int(.6 * len(flow)), int(.8 * len(flow))
    origin = np.arange(2016, len(flow) - 6 if external else second - 6)
    train = origin + 6 < first
    dev = origin >= (second if external else first)
    current, actual = flow[origin], flow[origin + 6]
    weekly_start, weekly = flow[origin - 2016], flow[origin + 6 - 2016]
    threshold = np.quantile((actual - current)[train], .9, axis=0)
    surge = (actual - current)[dev] > threshold
    gate = (weekly - weekly_start)[dev] > threshold
    phase = origin[:, None] * (2 * np.pi / 288)
    features = [flow[origin - lag] for lag in reversed(range(12))]
    features += [flow[origin + 6 - 288], weekly]
    features += [np.broadcast_to(f(phase), current.shape) for f in (np.sin, np.cos)]

    return origin, train, dev, actual, features, surge, gate, threshold


def main():
    origin, train, dev, actual, features, surge, gate, threshold = load_split()
    current, weekly = features[11], features[13]
    y = actual[dev]
    ridge_pred = ridge(features, actual, train, dev, .001)
    weekly_pred = weekly[dev]
    predictions = {
        "persistence": current[dev],
        "weekly": weekly_pred,
        "ridge": ridge_pred,
        "always_blend_50": .5 * (ridge_pred + weekly_pred),
        "conditional_blend_50": np.where(gate, .5 * (ridge_pred + weekly_pred), ridge_pred),
    }
    # Use complete dataset-aligned days for the paired block intervals.
    target_index = origin[dev] + 6
    days, blocks = np.unique(target_index // 288, return_inverse=True)
    complete = np.bincount(blocks)[blocks] == 288
    full_days, full_blocks = np.unique(days[blocks[complete]], return_inverse=True)
    draws = np.random.default_rng(42).integers(len(full_days), size=(5000, len(full_days)))
    groups = {"overall": np.ones_like(surge), "all_surge": surge,
              "nonrecurring_surge": surge & ~gate, "recurring_surge": surge & gate}
    report = {
        "evaluation": "PeMS08 development, previously inspected dataset",
        "train_n": int(train.sum() * actual.shape[1]),
        "bootstrap_complete_days": int(len(full_days)),
        "group_n": {name: int(mask.sum()) for name, mask in groups.items()},
        "scores": {},
    }
    for name, prediction in predictions.items():
        report["scores"][name] = {
            **score(prediction, y, surge),
            "group_mae": {group: float(np.abs(prediction - y)[mask].mean())
                          for group, mask in groups.items()},
            "paired_vs_ridge_full_days": {
                group: paired_summary(prediction[complete], ridge_pred[complete], y[complete],
                                      mask[complete], full_blocks, draws)
                for group, mask in groups.items()
            },
        }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUTPUT / "stage1_predictions.npz", origin_index=origin[dev],
                        target_index=target_index, actual=y, threshold=threshold,
                        surge=surge, weekly_gate=gate, **predictions)
    (OUTPUT / "stage1_metrics.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
