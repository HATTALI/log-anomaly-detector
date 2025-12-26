from __future__ import annotations

import os
import pandas as pd


def _precision_recall_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def main() -> None:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    anomalies_path = os.path.join(repo_root, "outputs", "anomalies.csv")
    labels_path = os.path.join(repo_root, "data", "hdfs_labels.csv")

    if not os.path.exists(anomalies_path):
        raise FileNotFoundError("Missing outputs/anomalies.csv. Run detect_anomalies.py first.")
    if not os.path.exists(labels_path):
        raise FileNotFoundError("Missing data/hdfs_labels.csv. Run prepare_hdfs_hf.py first.")

    anomalies = pd.read_csv(anomalies_path)
    labels = pd.read_csv(labels_path)

    merged = anomalies.merge(labels, on="line_number", how="inner")

    ks = [100, 500, 1000, 2000, 5000]

    print(f"Total rows evaluated: {len(merged)}")
    print(f"Total true anomalies in slice: {int(merged['anomaly'].sum())}\n")

    for k in ks:
        k = min(k, len(merged))
        topk = merged.head(k)

        tp = int(topk["anomaly"].sum())
        fp = k - tp

        rest = merged.iloc[k:]
        fn = int(rest["anomaly"].sum())

        precision, recall, f1 = _precision_recall_f1(tp, fp, fn)

        print(
            f"Top {k:5d} | TP={tp:5d} FP={fp:5d} FN={fn:5d} | "
            f"Precision={precision:.4f} Recall={recall:.4f} F1={f1:.4f}"
        )


if __name__ == "__main__":
    main()
