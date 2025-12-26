from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import List, Dict

import pandas as pd


SUSPICIOUS_KEYWORDS = [
    "error", "exception", "fail", "failed", "denied", "timeout", "refused",
    "unauthorized", "invalid",
    "warn", "warning", "fatal", "corrupt", "missing", "disconnected",
]


@dataclass
class LogFeatures:
    line_number: int
    raw_line: str

    # basic shape features
    char_count: int
    word_count: int
    digit_count: int
    special_char_count: int

    # keyword features
    keyword_hit_count: int
    has_error_keyword: int
    has_auth_keyword: int

    # log-template features (big win for logs)
    template: str
    template_freq: int


def _count_special_characters(text: str) -> int:
    return sum(1 for ch in text if not ch.isalnum() and not ch.isspace())


def _keyword_flags(lower_text: str) -> Dict[str, int]:
    error_terms = ["error", "exception", "timeout", "refused", "fatal", "corrupt", "warning", "warn"]
    auth_terms = ["failed", "denied", "unauthorized", "invalid"]

    has_error_keyword = 0
    for term in error_terms:
        if term in lower_text:
            has_error_keyword = 1
            break

    has_auth_keyword = 0
    for term in auth_terms:
        if term in lower_text:
            has_auth_keyword = 1
            break

    return {
        "has_error_keyword": has_error_keyword,
        "has_auth_keyword": has_auth_keyword,
    }


def _make_template(text: str) -> str:
    """
    Normalize variable parts (IDs, IPs, ports, numbers) so similar log messages map to same template.
    This helps anomaly detection a lot on HDFS logs.
    """
    t = text.lower()

    # block IDs
    t = re.sub(r"blk[_-]?-?\d+", "blk_<ID>", t)

    # IPs like 10.250.19.102
    t = re.sub(r"\b\d{1,3}(?:\.\d{1,3}){3}\b", "<IP>", t)

    # ports like :50010
    t = re.sub(r":\d+\b", ":<PORT>", t)

    # any remaining numbers
    t = re.sub(r"\b\d+\b", "<NUM>", t)

    # collapse extra spaces
    t = re.sub(r"\s+", " ", t).strip()
    return t


def extract_features(lines: List[str]) -> List[LogFeatures]:
    cleaned = [ln.strip() for ln in lines if ln.strip()]
    templates = [_make_template(ln) for ln in cleaned]

    template_counts: Dict[str, int] = {}
    for t in templates:
        template_counts[t] = template_counts.get(t, 0) + 1

    features: List[LogFeatures] = []

    for idx, clean_line in enumerate(cleaned, start=1):
        lower_line = clean_line.lower()

        words = re.findall(r"\S+", clean_line)
        digit_count = sum(ch.isdigit() for ch in clean_line)
        special_char_count = _count_special_characters(clean_line)

        keyword_hit_count = 0
        for kw in SUSPICIOUS_KEYWORDS:
            if kw in lower_line:
                keyword_hit_count += 1

        flags = _keyword_flags(lower_line)

        template = _make_template(clean_line)
        template_freq = template_counts.get(template, 0)

        features.append(
            LogFeatures(
                line_number=idx,
                raw_line=clean_line,
                char_count=len(clean_line),
                word_count=len(words),
                digit_count=digit_count,
                special_char_count=special_char_count,
                keyword_hit_count=keyword_hit_count,
                has_error_keyword=flags["has_error_keyword"],
                has_auth_keyword=flags["has_auth_keyword"],
                template=template,
                template_freq=template_freq,
            )
        )

    return features


def main() -> None:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(repo_root, "data", "hdfs_logs.txt")  # use HDFS dataset
    output_dir = os.path.join(repo_root, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Missing {data_path}. Run: python src/prepare_hdfs_hf.py first."
        )

    with open(data_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    feats = extract_features(lines)
    df = pd.DataFrame([x.__dict__ for x in feats])

    out_csv = os.path.join(output_dir, "features.csv")
    df.to_csv(out_csv, index=False)

    print(f"Loaded {len(df)} log lines")
    print(f"Saved features to: {out_csv}")
    print(df.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
