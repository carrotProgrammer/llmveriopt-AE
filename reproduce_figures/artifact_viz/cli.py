# -*- coding: utf-8 -*-
import logging
import yaml
import os
from .router import visualize

def main():
    # config file path (fixed for simplicity)
    config_file = os.path.join(
        os.path.dirname(__file__), "config", "config.yaml"
    )
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file not found: {config_file}")

    with open(config_file, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    models = cfg.get("models", [])
    metrics = cfg.get("metrics", [])
    dataset_path = cfg.get("dataset_path")
    log_level = cfg.get("log_level", "INFO").upper()
    output_dir = cfg.get("output_dir", "outputs")
    mode = str(cfg.get("mode", "bar")).strip().lower()

    logging.basicConfig(level=getattr(logging, log_level))

    # ensure relative to artifact/
    project_root = os.path.dirname(os.path.dirname(__file__))  # points to artifact/
    output_dir = os.path.join(project_root, output_dir)

    scenario = visualize(
        models=models,
        metrics=metrics,
        dataset_path=dataset_path,
        output_dir=output_dir,
        mode=mode,
    )
    logging.info("Plot mode: %s", mode)
    logging.info("Completed scenario: %s", scenario)

if __name__ == "__main__":
    main()
