"""Chronological 30-minute flow forecasting on PeMS04."""
import argparse
import json
from pathlib import Path

import numpy as np


def ridge(features, target, train, indices, penalty, offset=0):
    x = np.stack(features, axis=-1)
    fit = x[train].reshape(-1, x.shape[-1])
    mean = fit.mean(axis=0)
    scale = np.maximum(fit.std(axis=0), 1e-6)
    fit = (fit - mean) / scale
    y = target[train].ravel()
    weights = np.linalg.solve(
        fit.T @ fit + penalty * len(fit) * np.eye(fit.shape[1]),
        fit.T @ (y - y.mean()),
    )
    return np.maximum(((x[indices] - mean) / scale) @ weights + y.mean() + offset, 0)


def score(prediction, actual, surge):
    error = np.abs(prediction - actual)
    return {
        "mae": round(float(error.mean()), 4),
        "surge_mae": round(float(error[surge].mean()), 4) if surge.any() else None,
        "n": int(error.size),
        "surge_n": int(surge.sum()),
    }


def run(path):
    with np.load(path) as archive:
        flow = archive["data"][:, :, 0].astype(np.float64)
    if not np.isfinite(flow).all() or (flow < 0).any():
        raise ValueError("Flow must contain finite nonnegative counts.")
    origins = np.arange(2016, len(flow) - 6)
    train = origins + 6 < int(len(flow) * .6)
    dev = (origins >= int(len(flow) * .6)) & (origins + 6 < int(len(flow) * .8))
    test = origins >= int(len(flow) * .8)
    target = flow[origins + 6]
    current = flow[origins]
    surge = (target - current) > np.quantile((target - current)[train], .9, axis=0)
    weekly = flow[origins + 6 - 2016]
    daily = flow[origins + 6 - 288]
    recent = [flow[origins - lag] for lag in (1, 2, 5, 11)]
    phase = origins[:, None] * (2 * np.pi / 288)
    clock = [np.broadcast_to(f(phase), current.shape) for f in (np.sin, np.cos)]
    stages = {
        "01_weekly": [current, weekly],
        "02_daily": [current, weekly, daily],
        "03_recent_history": [current, weekly, daily, *recent],
        "04_time_of_day": [current, weekly, daily, *recent, *clock],
        "05_sensor_adjustment": [
            current, weekly, daily, *recent, *clock,
            np.broadcast_to(target[train].mean(axis=0), current.shape),
        ],
    }
    scores = {"persistence": score(current[dev], target[dev], surge[dev]),
              "weekly_naive": score(weekly[dev], target[dev], surge[dev])}
    for name, features in stages.items():
        prediction = ridge(features, target, train, dev, .001)
        scores[name] = score(prediction, target[dev], surge[dev])
    selected = min(scores, key=lambda name: scores[name]["mae"])
    if selected in stages:
        final = ridge(stages[selected], target, train, test, .001)
    else:
        final = (current if selected == "persistence" else weekly)[test]
    return {
        "shape": list(flow.shape),
        "origin_counts": {name: int(mask.sum()) for name, mask in
                          (("train", train), ("development", dev), ("test", test))},
        "development": scores,
        "selected_by_dev_mae": selected,
        "test": {"selected": score(final, target[test], surge[test]),
                 "persistence": score(current[test], target[test], surge[test]),
                 "weekly_naive": score(weekly[test], target[test], surge[test])},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.dataset), indent=2))
