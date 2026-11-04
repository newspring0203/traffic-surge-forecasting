"""Compare five model families on the locked PeMS08 development sample."""
import argparse
import json
from time import perf_counter

import numpy as np
import torch
from torch import nn
import xgboost as xgb

from compare_models import Forecast
from evaluate_stage1 import OUTPUT, load_split, paired_summary
from experiment import ridge


def train_network(model, fit_x, fit_y, dev_x, epochs, mean, scale):
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002)
    losses = []
    for epoch in range(epochs):
        model.train()
        total = 0.
        for batch in torch.randperm(len(fit_x)).split(2048):
            loss = (model(fit_x[batch]) - fit_y[batch]).square().mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(batch)
        losses.append(total / len(fit_x))
        print(f"{type(model).__name__} epoch {epoch + 1}: {losses[-1]:.6f}", flush=True)
    model.eval()
    with torch.no_grad():
        values = torch.cat([model(chunk) for chunk in dev_x.split(4096)]).numpy()
    return np.maximum(values * scale + mean, 0), losses


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--external-pems03", action="store_true")
    external = parser.parse_args().external_pems03
    stage = "stage5" if external else "stage2"
    dataset = "data/PEMS03.npz" if external else "data/PEMS08.npz"
    torch.set_num_threads(4)
    origin, train, dev, actual, features, surge, gate, threshold = load_split(dataset, external)
    target_index = origin[dev] + 6
    y = actual[dev]
    ridge_pred = ridge(features, actual, train, dev, .001)
    weekly = features[13][dev]
    predictions = {
        "weekly": weekly,
        "ridge": ridge_pred,
        "always_blend_50": .5 * (ridge_pred + weekly),
        "conditional_blend_50": np.where(gate, .5 * (ridge_pred + weekly), ridge_pred),
    }
    if not external:
        with np.load(OUTPUT / "stage1_predictions.npz") as baseline:
            np.testing.assert_array_equal(target_index, baseline["target_index"])
            np.testing.assert_array_equal(y, baseline["actual"])
            np.testing.assert_allclose(ridge_pred, baseline["ridge"], rtol=0, atol=1e-7)

    matrix = np.stack(features, axis=-1).astype(np.float32)
    fit_x = matrix[train].reshape(-1, 16)
    fit_y = actual[train].ravel().astype(np.float32)
    dev_x = matrix[dev].reshape(-1, 16)
    del matrix
    config = {"seed": 42, "epochs": 5, "objective": "squared_error",
              "train_n": len(fit_y), "evaluation": "PeMS03 last 20%" if external else "PeMS08 development",
              "models": {}}
    start = perf_counter()
    tree = xgb.train({"max_depth": 5, "eta": .05, "tree_method": "hist",
                      "objective": "reg:squarederror", "nthread": 4, "seed": 42},
                     xgb.DMatrix(fit_x, label=fit_y, nthread=4), num_boost_round=200)
    predictions["xgboost"] = np.maximum(tree.inplace_predict(dev_x), 0).reshape(y.shape)
    config["models"]["xgboost"] = {"seconds": perf_counter() - start,
                                  "trees": 200, "max_depth": 5, "learning_rate": .05}
    if external:
        with np.load(dataset) as archive:
            raw = archive["data"]
            flow = raw[:, :, 0] if raw.ndim == 3 else raw
            prior_rise = features[13][train] - flow[origin[train] - 2016]
        novel = ((actual[train] - features[11][train]) > threshold) & (prior_rise <= threshold)
        weights = np.where(novel.ravel(), 1.5, 1.).astype(np.float32)
        training = xgb.DMatrix(fit_x, label=fit_y, weight=weights / weights.mean(), nthread=4)
        weighted = xgb.train({"max_depth": 5, "eta": .05, "tree_method": "hist",
                              "objective": "reg:squarederror", "nthread": 4, "seed": 42},
                             training, num_boost_round=200)
        predictions["xgboost_weight_1.5"] = np.maximum(weighted.inplace_predict(dev_x), 0).reshape(y.shape)
        config["models"]["xgboost_weight_1.5"] = {"weight": 1.5, "upweighted_n": int(novel.sum())}
    del features
    print("XGBoost complete", flush=True)

    mean = float(fit_x[:, 11].mean())
    scale = float(fit_x[:, 11].std())
    for inputs in (fit_x, dev_x):
        inputs[:, :14] = (inputs[:, :14] - mean) / scale
    fit_x, dev_x = torch.from_numpy(fit_x), torch.from_numpy(dev_x)
    fit_y = torch.from_numpy((fit_y - mean) / scale)
    for name in ("mlp", "lstm", "transformer"):
        torch.manual_seed(42)
        model = (nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 1),
                               nn.Flatten(0)) if name == "mlp" else Forecast(name))
        start = perf_counter()
        forecast, losses = train_network(model, fit_x, fit_y, dev_x, 5, mean, scale)
        predictions[name] = forecast.reshape(y.shape)
        config["models"][name] = {"seconds": perf_counter() - start,
                                  "parameters": sum(p.numel() for p in model.parameters()),
                                  "training_mse_standardized": losses}

    groups = {"overall": np.ones_like(surge), "all_surge": surge,
              "nonrecurring_surge": surge & ~gate, "recurring_surge": surge & gate}
    _, blocks = np.unique(target_index // 288, return_inverse=True)
    complete = np.bincount(blocks)[blocks] == 288
    days, full_blocks = np.unique(blocks[complete], return_inverse=True)
    draws = np.random.default_rng(42).integers(len(days), size=(5000, len(days)))
    reference = predictions["always_blend_50"]
    config["group_n"] = {group: int(mask.sum()) for group, mask in groups.items()}
    config["bootstrap_complete_days"] = len(days)
    config["scores"] = {}
    for name, pred in predictions.items():
        assert np.isfinite(pred).all() and pred.shape == y.shape
        config["scores"][name] = {
            "mae": {group: float(np.abs(pred - y)[mask].mean()) for group, mask in groups.items()},
            "paired_vs_always_blend_50_full_days": {
                group: paired_summary(pred[complete], reference[complete], y[complete],
                                      mask[complete], full_blocks, draws)
                for group, mask in groups.items()
            },
        }
    np.savez_compressed(OUTPUT / f"{stage}_predictions.npz", target_index=target_index,
                        actual=y, surge=surge, weekly_gate=gate, **predictions)
    (OUTPUT / f"{stage}_metrics.json").write_text(json.dumps(config, indent=2) + "\n")


if __name__ == "__main__":
    main()
