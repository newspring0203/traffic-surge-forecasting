"""Matched exploratory 30-minute forecasts on PeMS08."""
import json
import numpy as np
import torch
from torch import nn
from experiment import ridge, score


class Forecast(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind
        if kind == "lstm":
            self.encoder = nn.LSTM(1, 16, batch_first=True)
        else:
            self.project = nn.Linear(1, 16)
            self.position = nn.Parameter(torch.randn(1, 12, 16) * .02)
            layer = nn.TransformerEncoderLayer(16, 2, 32, dropout=0, batch_first=True)
            self.encoder = nn.TransformerEncoder(layer, 1)
        self.head = nn.Sequential(nn.Linear(20, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, x):
        history = x[:, :12, None]
        if self.kind == "lstm":
            encoded, _ = self.encoder(history)
        else:
            encoded = self.encoder(self.project(history) + self.position)
        return self.head(torch.cat([encoded[:, -1], x[:, 12:]], dim=1)).squeeze(1)


def main():
    torch.set_num_threads(4)
    with np.load("data/PEMS08.npz") as data:
        flow = data["data"][:, :, 0].astype(np.float32)
    origin = np.arange(2016, len(flow) - 6)
    first, second = int(.6 * len(flow)), int(.8 * len(flow))
    train = origin + 6 < first
    dev = (origin >= first) & (origin + 6 < second)
    test = origin >= second
    current, actual = flow[origin], flow[origin + 6]
    weekly = flow[origin + 6 - 2016]
    threshold = np.quantile((actual - current)[train], .9, axis=0)
    surge = (actual - current)[test] > threshold
    novel = surge & ((weekly - flow[origin - 2016])[test] <= threshold)
    phase = origin[:, None] * (2 * np.pi / 288)
    features = [flow[origin - lag] for lag in reversed(range(12))]
    features += [flow[origin + 6 - 288], weekly]
    features += [np.broadcast_to(f(phase), current.shape) for f in (np.sin, np.cos)]

    def metrics(prediction):
        return {**score(prediction, actual[test], surge),
                "nonrecurring_surge_mae": float(np.abs(prediction - actual[test])[novel].mean())}

    report = {"weekly": metrics(weekly[test])}
    # Equal deterministic subsampling for all trained models keeps the CPU run small.
    fit = train & (origin % 6 == 0)
    report["ridge"] = metrics(ridge(features, actual, fit, test, .001))
    x = np.stack(features, axis=-1).astype(np.float32)
    mean = float(current[fit].mean())
    scale = float(current[fit].std())

    def tensors(mask):
        inputs = x[mask].reshape(-1, 16).copy()
        inputs[:, :14] = (inputs[:, :14] - mean) / scale
        return torch.from_numpy(inputs), torch.from_numpy((actual[mask].ravel() - mean) / scale)

    fit_x, fit_y = tensors(fit)
    dev_x, dev_y = tensors(dev & (origin % 6 == 0))
    test_x, _ = tensors(test)
    for kind in ("lstm", "transformer"):
        torch.manual_seed(42)
        model = Forecast(kind)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.002)
        best_loss = float("inf")
        for epoch in range(5):
            model.train()
            for batch in torch.randperm(len(fit_x)).split(1024):
                loss = (model(fit_x[batch]) - fit_y[batch]).abs().mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            model.eval()
            with torch.no_grad():
                loss = sum((model(chunk) - expected).abs().sum().item()
                           for chunk, expected in zip(dev_x.split(2048), dev_y.split(2048))) / len(dev_y)
            if loss < best_loss:
                best_loss = loss
                best_epoch = epoch + 1
                best = {key: value.clone() for key, value in model.state_dict().items()}
        model.load_state_dict(best)
        with torch.no_grad():
            output = torch.cat([model(chunk) for chunk in test_x.split(2048)]).numpy()
        report[kind] = {**metrics(np.maximum(output.reshape(actual[test].shape) * scale + mean, 0)),
                        "selected_epoch": best_epoch}
    print(json.dumps({"train_n": len(fit_x), "novel_n": int(novel.sum()), "scores": report}, indent=2))


if __name__ == "__main__":
    main()
