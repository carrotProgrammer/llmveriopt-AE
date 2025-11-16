# -*- coding: utf-8 -*-
from typing import Iterable, Union, List
from .config import VizConfig
import logging
from .constants import Scenario, NON_CORR_METRICS, VIOLIN_TARGET_MODELS, BASELINE_MODELS
from .data_loader import load_dataset
from .renderers import (
    render_table_I,
    render_table_III,
    render_fig5,
    render_tableIII_and_fig7,
    render_violin,
)

def decide_scenario(models: List[str], metrics: List[str]) -> str:
    n_models = len(models)
    metric_set = set(metrics)

    if n_models == 1:
        # A: exactly {'correctness'}
        if metric_set == {"correctness"}:
            return Scenario.A_TABLE_I
        # B: contains any non-correctness metric(s)
        if metric_set & NON_CORR_METRICS:
            return Scenario.B_TABLE_III
        # Defensive fallback
        raise ValueError("Single-model configuration does not match any scenario.")
    else:
        # multiple models
        if len(metric_set) == 1:
            return Scenario.C_FIG5
        else:
            return Scenario.D_TABLEIII_FIG7

def visualize(
    models,
    metrics,
    dataset_path: str,
    output_dir: str = "outputs",
    mode: str = "bar", 
) -> str:
    cfg = VizConfig.from_raw(models=models, metrics=metrics, dataset_path=dataset_path,mode=mode)
    data = load_dataset(cfg.dataset_path)
    scenario = decide_scenario(cfg.models, cfg.metrics)

    if cfg.mode == "violin":
        targets = [m for m in cfg.models if m in VIOLIN_TARGET_MODELS]
        ignored = [m for m in cfg.models if m in BASELINE_MODELS]
        if ignored:
            logging.info("Violin mode: ignoring baselines in models list: %s", ignored)

        if not targets:
            logging.warning(
                "Violin mode: no target models in config; expected one of %s",
                sorted(VIOLIN_TARGET_MODELS),
            )
            return scenario

        for m in targets:
            paths = render_violin(data, m, cfg.metrics, output_dir)
            print(f"Saved Violin for {m} to {paths}")

        return scenario

    if scenario == Scenario.A_TABLE_I:
        path = render_table_I(data, cfg.models[0], output_dir)
        print(f"Saved Table I to {path}")

    elif scenario == Scenario.B_TABLE_III:
        path = render_table_III(data, cfg.models[0], cfg.metrics, output_dir)
        print(f"Saved Table III to {path}")

    elif scenario == Scenario.C_FIG5:
        path = render_fig5(data, cfg.models, cfg.metrics[0], output_dir)
        print(f"Saved Fig5 to {path}")

    elif scenario == Scenario.D_TABLEIII_FIG7:
        paths = render_tableIII_and_fig7(data, cfg.models, cfg.metrics, output_dir)
        print(f"Saved Table III (extended) + Fig7 to {paths}")

    return scenario
