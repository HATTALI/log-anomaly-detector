from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import List, Dict

import pandas as pd

SUSPICIOUS_KEYWORDS = [
    "error",
    "fail",
    "failed",
    "denied",
    "timeout",
    "exception",
    "refused",
    "unauthorized",
    "invalid",
]

@dataclass
class LogFeatures:
    line_number: int
    raw_line: str
    char_count: int
    word_count: int
    digit_count: int
    special_char_count: int
    keyword_hit_count: int
    has_error_keyword: int
    has_auth_keyword: int


def _count_special_characters(text: str) -> int:
    # Count characters that are not letters/digits/whitespace
    return sum(1 for ch in text if not ch.isalnum() and not ch.isspace())

def _keyword_flags(lower_text: str) -> Dict[str, int]:
    # Two simple groups: "error-ish" and "auth-ish"
    error_terms = ["error", "exception", "timeout", "refused"]
    auth_terms = ["failed", "denied", "unauthorized", "invalid"]

    has_error_keyword = int(any(term in lower_text for term in error_terms))
    has_auth_keyword = int(any(term in lower_text for term in auth_terms))

    return {
        "has_error_keyword": has_error_keyword,
        "has_auth_keyword": has_auth_keyword,
    }

def extract_features(lines: List[str]) -> List[LogFeatures]:
    features: List[LogFeatures] = []

    for idx, line in enumerate(lines, start=1):
        clean_line = line.strip()
        lower_line = clean_line.lower()

        words = re.findall(r"\S+", clean_line)
        digits = sum(ch.isdigit() for ch in clean_line)
        special_chars = _count_special_characters(clean_line)

        keyword_hit_count = sum(1 for kw in SUSPICIOUS_KEYWORDS if kw in lower_line)
        flags = _keyword_flags(lower_line)

        features.append(
            LogFeatures(
                line_number=idx,
                raw_line=clean_line,
                char_count=len(clean_line),
                word_count=len(words),
                digit_count=digits,
                special_char_count=special_chars,
                keyword_hit_count=keyword_hit_count,
                has_error_keyword=flags["has_error_keyword"],
                has_auth_keyword=flags["has_auth_keyword"],
            )
        )

    return features


def main() -> None:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(repo_root, "data", "sample_logs.txt")
    output_dir = os.path.join(repo_root, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    with open(data_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    feats = extract_features(lines)
    df = pd.DataFrame([f.__dict__ for f in feats])

    out_csv = os.path.join(output_dir, "features.csv")
    df.to_csv(out_csv, index=False)

    # quick sanity print
    print(f"Loaded {len(lines)} log lines")
    print(f"Saved features to: {out_csv}")
    print(df.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
