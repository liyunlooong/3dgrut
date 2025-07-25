from typing import Dict, Any

import hydra
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining

import torch
from threedgrut.trainer import Trainer3DGRUT
from ray.air import session


def tune_train(config: Dict[str, Any]):
    """Wrapper to launch training inside Ray Tune."""
    overrides = []
    for key, value in config.items():
        if key in {"training_iteration", "reporter"}:
            continue
        overrides.append(f"{key}={value}")

    with hydra.initialize_config_dir("configs"):
        cfg = hydra.compose(config_name="apps/colmap_3dgrt.yaml", overrides=overrides)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def report(metrics):
        session.report(metrics)

    trainer = Trainer3DGRUT(cfg, device=device, report_hook=report)
    trainer.run_training()


def main():
    scheduler = ASHAScheduler(metric="psnr", mode="max")
    pbt = PopulationBasedTraining(
        time_attr="training_iteration",
        metric="psnr",
        mode="max",
        perturbation_interval=5,
        hyperparam_mutations={
            "optimizer.params.positions.lr": tune.loguniform(1e-5, 5e-4),
            "optimizer.params.density.lr": tune.loguniform(1e-3, 0.1),
        },
    )

    reporter = CLIReporter(metric_columns=["psnr"])

    search_space = {
        "path": "/data14/yunlong.li.2507/mipnerf360_dataset/bonsai",
        "out_dir": "runs",
        "experiment_name": "bonsai_3dgrt_tune",
        "dataset.downsample_factor": 2,
        "optimizer.type": "sghmc",
        "optimizer.params.positions.lr": tune.loguniform(1e-5, 5e-4),
        "optimizer.params.density.lr": tune.loguniform(1e-3, 0.1),
    }

    tuner = tune.Tuner(
        tune.with_parameters(tune_train),
        param_space=search_space,
        tune_config=tune.TuneConfig(num_samples=4, scheduler=pbt),
        run_config=tune.RunConfig(progress_reporter=reporter),
    )
    tuner.fit()


if __name__ == "__main__":
    main()
