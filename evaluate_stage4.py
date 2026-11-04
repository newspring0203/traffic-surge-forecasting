"""Test training weights for nonrecurring surges on PeMS08 development."""
import json

import numpy as np
import xgboost as xgb

from evaluate_stage1 import OUTPUT, load_split, paired_summary


def main():
    origin, train, dev, actual, features, surge, gate, threshold = load_split()
    target_index = origin[dev] + 6
    y = actual[dev]
    with np.load(OUTPUT / "stage2_predictions.npz") as previous:
        for key, expected in (("target_index", target_index), ("actual", y),
                              ("surge", surge), ("weekly_gate", gate)):
            np.testing.assert_array_equal(previous[key], expected)
        baseline = previous["xgboost"]
        blend = previous["always_blend_50"]

    with np.load("data/PEMS08.npz") as archive:
        flow = archive["data"][:, :, 0]
        prior_rise = features[13][train] - flow[origin[train] - 2016]
    novel_train = ((actual[train] - features[11][train]) > threshold) & (prior_rise <= threshold)
    matrix = np.stack(features, axis=-1).astype(np.float32)
    training = xgb.DMatrix(matrix[train].reshape(-1, 16),
                           label=actual[train].ravel(), nthread=4)
    evaluation = matrix[dev].reshape(-1, 16)
    params = {"max_depth": 5, "eta": .05, "tree_method": "hist",
              "objective": "reg:squarederror", "nthread": 4, "seed": 42}
    _, blocks = np.unique(target_index // 288, return_inverse=True)
    complete = np.bincount(blocks)[blocks] == 288
    days, full_blocks = np.unique(blocks[complete], return_inverse=True)
    draws = np.random.default_rng(42).integers(len(days), size=(5000, len(days)))
    groups = {"overall": np.ones_like(surge), "all_surge": surge,
              "nonrecurring_surge": surge & ~gate}
    predictions = {"xgboost": baseline, "always_blend_50": blend}
    for weight in (1.5, 2.0, 3.0):
        weights = np.where(novel_train.ravel(), weight, 1.).astype(np.float32)
        training.set_weight(weights / weights.mean())
        model = xgb.train(params, training, num_boost_round=200)
        pred = np.maximum(model.inplace_predict(evaluation), 0).reshape(y.shape)
        assert np.isfinite(pred).all()
        predictions[f"weight_{weight:g}"] = pred
        print(f"weight {weight:g} complete", flush=True)

    scores = {}
    for name, pred in predictions.items():
        scores[name] = {
            group: {
                "mae": float(np.abs(pred - y)[mask].mean()),
                "underprediction_rate": float((pred[mask] < y[mask]).mean()),
                **{"paired_vs_xgboost_full_days": paired_summary(
                    pred[complete], baseline[complete], y[complete],
                    mask[complete], full_blocks, draws)},
            }
            for group, mask in groups.items()
        }
    np.savez_compressed(OUTPUT / "stage4_predictions.npz", target_index=target_index,
                        actual=y, surge=surge, weekly_gate=gate, **predictions)
    report = {"train_n": novel_train.size, "upweighted_train_n": int(novel_train.sum()),
              "bootstrap_complete_days": len(days), "scores": scores}
    (OUTPUT / "stage4_metrics.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
