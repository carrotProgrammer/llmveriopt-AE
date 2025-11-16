# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, List, Union
import os
import logging

from .constants import ALLOWED_MODELS, ALLOWED_METRICS,ALLOWED_PLOT_MODES

@dataclass
class VizConfig:
    models: List[str]
    metrics: List[str]
    dataset_path: str  # e.g., /workspace/artifact/statistic_table
    mode: str = "bar" # （bar | violin）

    @staticmethod
    def from_raw(
        models: Union[str, Iterable[str]],
        metrics: Union[str, Iterable[str]],
        dataset_path: str,
        mode: str = "bar",
    ) -> "VizConfig":
        def _to_list(x: Union[str, Iterable[str]]) -> List[str]:
            if isinstance(x, str):
                return [x.strip()]
            return [str(i).strip() for i in x]

        cfg = VizConfig(
            models=_to_list(models),
            metrics=_to_list(metrics),
            dataset_path=os.path.abspath(dataset_path),
            mode=(mode or "bar").strip().lower(),
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        # Check models
        bad_models = [m for m in self.models if m not in ALLOWED_MODELS]
        if bad_models:
            raise ValueError(
                f"Unknown model(s): {bad_models}. "
                f"Allowed: {sorted(ALLOWED_MODELS)}"
            )

        # Check metrics
        bad_metrics = [m for m in self.metrics if m not in ALLOWED_METRICS]
        if bad_metrics:
            raise ValueError(
                f"Unknown metric(s): {bad_metrics}. "
                f"Allowed: {sorted(ALLOWED_METRICS)}"
            )

        # Check dataset path
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"dataset_path does not exist: {self.dataset_path}")

        # Check mode
        if self.mode not in ALLOWED_PLOT_MODES:
            raise ValueError(
                f"Unknown mode: {self.mode}. Allowed: {sorted(ALLOWED_PLOT_MODES)}"
            )

        # Warn about not-yet-implemented metric
        if "real_latency" in self.metrics:
            logging.warning(
                "Metric 'real_latency' is not implemented yet. "
                "It will be skipped or shown as N/A in placeholders."
            )
