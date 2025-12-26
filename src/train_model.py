from __future__ import annotations

import os
import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest


FEATURE_COLUMNS = [
    "char_count",
    "word_count",
    "digit_count",
    "special_char_count",
    "keyword_hit_count",
    "has_error_keyword",
    "has_auth_keyword",
    "template_freq",   # IMPORTANT: new feature
]


def main() -> None:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(repo_root, "outputs")
    features_path = os.path.join(output_dir, "features.csv")
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(features_path):
        raise FileNotFoundError(
            f"Could not find {features_path}. Run: python src/make_features.py first."
        )

    df = pd.read_csv(features_path)

    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns in features.csv: {missing}")

    X = df[FEATURE_COLUMNS].copy()

    model = IsolationForest(
        n_estimators=200,
        contamination=0.05,  # estimate of anomaly rate; we’ll evaluate with top-K
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X)

    model_path = os.path.join(output_dir, "model.pkl")
    joblib.dump(
        {
            "model": model,
            "feature_columns": FEATURE_COLUMNS,
        },
        model_path,
    )

    print("Training complete ✅")
    print(f"Loaded rows: {len(df)}")
    print(f"Saved model to: {model_path}")
    print(f"Using features: {FEATURE_COLUMNS}")


if __name__ == "__main__":
    main()
