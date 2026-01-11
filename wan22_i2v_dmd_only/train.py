import argparse
import os
from omegaconf import OmegaConf
import mlflow

from trainer import ScoreDistillationTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--no_save", action="store_true")
    parser.add_argument("--no_visualize", action="store_true")
    parser.add_argument("--logdir", type=str, default="", help="Path to the directory to save logs")
    parser.add_argument("--mlflow-tracking-uri", type=str, default="", help="MLflow tracking URI")
    parser.add_argument("--mlflow-experiment", type=str, default="", help="MLflow experiment name")
    parser.add_argument("--mlflow-run-name", type=str, default="", help="MLflow run name")
    parser.add_argument("--disable-mlflow", action="store_true")
    parser.add_argument("--disable-wandb", action="store_true", help="Deprecated: use --disable-mlflow")

    args = parser.parse_args()

    config = OmegaConf.load(args.config_path)
    default_config = OmegaConf.load("configs/default_config.yaml")
    config = OmegaConf.merge(default_config, config)
    config.no_save = args.no_save
    config.no_visualize = args.no_visualize

    # get the filename of config_path
    config_name = os.path.basename(args.config_path).split(".")[0]
    config.config_name = config_name
    config.logdir = args.logdir
    config.disable_mlflow = args.disable_mlflow or args.disable_wandb
    if args.mlflow_tracking_uri:
        config.mlflow_tracking_uri = args.mlflow_tracking_uri
    if args.mlflow_experiment:
        config.mlflow_experiment = args.mlflow_experiment
    if args.mlflow_run_name:
        config.mlflow_run_name = args.mlflow_run_name

    if config.trainer == "score_distillation":
        trainer = ScoreDistillationTrainer(config)
    else:
        raise ValueError("Only score_distillation trainer is supported in this project.")
    trainer.train()

    if not getattr(config, "disable_mlflow", False):
        mlflow.end_run()


if __name__ == "__main__":
    main()
