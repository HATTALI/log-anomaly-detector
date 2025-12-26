from __future__ import annotations

import os
import pandas as pd
from datasets import load_dataset


def main() -> None:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(repo_root, "data")
    os.makedirs(data_dir, exist_ok=True)

    # Download dataset (cached automatically)
    ds = load_dataset("logfit-project/HDFS_v1", split="train")

    # Keep a manageable slice for local work
    max_rows = 50000
    ds_small = ds.select(range(min(max_rows, len(ds))))
    df = ds_small.to_pandas()

    logs_path = os.path.join(data_dir, "hdfs_logs.txt")
    labels_path = os.path.join(data_dir, "hdfs_labels.csv")

    with open(logs_path, "w", encoding="utf-8") as f:
        for msg in df["content"].astype(str).tolist():
            f.write(msg.replace("\n", " ").strip() + "\n")

    labels_df = pd.DataFrame(
        {
            "line_number": list(range(1, len(df) + 1)),
            "anomaly": df["anomaly"].astype(int).tolist(),
        }
    )
    labels_df.to_csv(labels_path, index=False)

    print("HDFS prep complete ✅")
    print(f"Wrote logs to: {logs_path}")
    print(f"Wrote labels to: {labels_path}")
    print(f"Rows: {len(df)}")


if __name__ == "__main__":
    main()
