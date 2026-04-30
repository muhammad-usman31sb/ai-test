from __future__ import annotations

import csv
import os
from typing import Iterable, Mapping


def ensure_parent_dir(file_path: str) -> None:
    parent = os.path.dirname(file_path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def append_csv_rows(file_path: str, rows: Iterable[Mapping[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        return

    ensure_parent_dir(file_path)

    fieldnames = list(rows[0].keys())
    file_exists = os.path.exists(file_path)

    with open(file_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)
