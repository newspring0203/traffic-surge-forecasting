"""Compare paired XGBoost and blend errors on PeMS08 development surges."""
import json

import numpy as np

from evaluate_stage1 import OUTPUT


def main():
    with np.load(OUTPUT / "stage2_predictions.npz") as data:
        mask = data["surge"] & ~data["weekly_gate"]
        rows, sensors = np.where(mask)
        targets = data["target_index"][rows]
        actual = data["actual"][mask]
        xgb = data["xgboost"][mask] - actual
        blend = data["always_blend_50"][mask] - actual

    under_difference = np.maximum(-xgb, 0) - np.maximum(-blend, 0)
    over_difference = np.maximum(xgb, 0) - np.maximum(blend, 0)
    difference = under_difference + over_difference

    def group_summary(ids):
        groups = []
        for key in np.unique(ids):
            selected = ids == key
            groups.append({
                "id": int(key),
                "n": int(selected.sum()),
                "mae_difference": float(difference[selected].mean()),
                "total_error_difference": float(difference[selected].sum()),
                "underprediction_difference": float(under_difference[selected].mean()),
            })
        return sorted(groups, key=lambda group: group["total_error_difference"], reverse=True)

    report = {
        "n": len(actual),
        "mae_difference": float(difference.mean()),
        "underprediction_difference": float(under_difference.mean()),
        "overprediction_difference": float(over_difference.mean()),
        "xgboost_underprediction_rate": float((xgb < 0).mean()),
        "blend_underprediction_rate": float((blend < 0).mean()),
        "days": group_summary(targets // 288),
        "sensors": group_summary(sensors),
    }
    (OUTPUT / "stage3_error_analysis.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
