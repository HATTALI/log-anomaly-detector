from __future__ import annotations

import os
import joblib
import pandas as pd


def main() -> None:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(repo_root, "outputs")

    model_path = os.path.join(output_dir, "model.pkl")
    features_path = os.path.join(output_dir, "features.csv")
    anomalies_path = os.path.join(output_dir, "anomalies.csv")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Could not find {model_path}. Run: python src/train_model.py first."
        )
    if not os.path.exists(features_path):
        raise FileNotFoundError(
            f"Could not find {features_path}. Run: python src/make_features.py first."
        )

    saved = joblib.load(model_path)
    model = saved["model"]
    feature_columns = saved["feature_columns"]

    df = pd.read_csv(features_path)

    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        raise ValueError(f"features.csv missing columns needed by model: {missing}")

    X = df[feature_columns].copy()

    # score_samples: higher = more normal; lower = more anomalous
    raw_score = model.score_samples(X)

    # make higher = more suspicious (easier to read)
    anomaly_score = -raw_score

    result = df.copy()
    result["raw_score"] = raw_score
    result["anomaly_score"] = anomaly_score

    result = result.sort_values(by="anomaly_score", ascending=False)

    os.makedirs(output_dir, exist_ok=True)
    result.to_csv(anomalies_path, index=False)

    print("Detection complete ✅")
    print(f"Saved anomalies to: {anomalies_path}")
    print("\nTop 10 most suspicious log lines:\n")
    top = result.head(10)[["line_number", "anomaly_score", "raw_line"]]
    print(top.to_string(index=False))


if __name__ == "__main__":
    main()
