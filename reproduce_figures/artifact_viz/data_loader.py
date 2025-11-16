# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Any, List
import os
import glob
import logging

def _is_hf_datasets_dir(path: str) -> bool:
    # Heuristics for load_from_disk-compatible dirs
    return os.path.isdir(path) and any(
        os.path.exists(os.path.join(path, marker))
        for marker in (
            "dataset_info.json",  # new style
            "state.json",         # HF datasets 2.x
            "arrow",              # sometimes cached shards live here
            "data-00000-of-00001.arrow",  # single-shard hint
        )
    )

def _load_hf_datasets(path: str):
    try:
        from datasets import load_from_disk, Dataset, DatasetDict
    except Exception as e:
        logging.debug("HuggingFace datasets not available: %s", e)
        return None

    try:
        ds = load_from_disk(path)
    except Exception as e:
        logging.warning("Failed to load HuggingFace dataset from %s: %s", path, e)
        return None

    import pandas as pd
    if hasattr(ds, "keys"):  # DatasetDict
        frames: List[pd.DataFrame] = []
        for split_name, split in ds.items():  # type: ignore[attr-defined]
            frames.append(split.to_pandas().assign(split=split_name))
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    else:  # Dataset
        return ds.to_pandas()

def _load_csv_any(path: str):
    import pandas as pd

    if os.path.isfile(path) and path.lower().endswith(".csv"):
        return pd.read_csv(path)

    if os.path.isdir(path):
        csvs = sorted(glob.glob(os.path.join(path, "*.csv")))
        if not csvs:
            raise FileNotFoundError(
                f"No CSV files found in directory: {path}. "
                "If you intended to use a HuggingFace dataset, ensure the folder "
                "was saved with datasets.save_to_disk(...)."
            )
        frames = [pd.read_csv(p) for p in csvs]
        return pd.concat(frames, ignore_index=True)

    raise FileNotFoundError(f"Path not found or not CSV: {path}")

def load_dataset(dataset_path: str) -> Any:
    """
    Default: try loading as HuggingFace Datasets (load_from_disk).
    Fallback: CSV (single file or directory of CSVs).
    Returns a pandas.DataFrame for uniform downstream use.
    """
    dataset_path = os.path.abspath(dataset_path)

    if _is_hf_datasets_dir(dataset_path):
        df = _load_hf_datasets(dataset_path)
        if df is not None:
            return df

    try:
        return _load_csv_any(dataset_path)
    except Exception as e:
        logging.error("Failed to load dataset from %s: %s", dataset_path, e)
        return {"_placeholder": True, "dataset_path": dataset_path, "error": str(e)}
