"""LLM-driven hyperparameter tuning for 3DGRUT."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Dict, Any

import yaml
from omegaconf import OmegaConf
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

try:
    import openai
except ImportError:  # pragma: no cover - openai might not be installed
    openai = None


class LLMAutoTuner:
    """Automatically tune hyperparameters via an LLM."""

    def __init__(
        self,
        base_config: Path,
        out_dir: Path,
        rounds: int = 3,
        model: str = "gpt-4",
        api_key: str | None = None,
        overrides: Dict[str, Any] | None = None,
    ) -> None:
        self.base_config = Path(base_config)
        self.out_dir = Path(out_dir)
        self.rounds = rounds
        self.model = model
        self.conf = OmegaConf.load(self.base_config)
        self.overrides: Dict[str, Any] = overrides or {}
        # root directory that contains the repository configs
        self.config_root = self.base_config.parents[1]
        if openai is not None:
            key = api_key or os.getenv("OPENAI_API_KEY")
            if not key:
                raise ValueError(
                    "OpenAI API key not provided. Set OPENAI_API_KEY env var or pass --api-key"
                )
            openai.api_key = key

    def _run_training(self, config: Path) -> Path:
        """Run a single training round and return the log directory."""
        cmd = [
            "python",
            "train.py",
            "--config-path",
            str(config.parent),
            "--config-name",
            config.name,
            f"hydra.searchpath=[file://{self.config_root.resolve()}]",
        ]
        env = os.environ.copy()
        env["HYDRA_FULL_ERROR"] = "1"
        subprocess.run(cmd, check=True, env=env)
        # Training logs are written under {conf.out_dir}/{conf.experiment_name}/<run>
        conf = OmegaConf.load(config)
        out_dir = Path(conf.out_dir) / conf.experiment_name
        last_run = sorted(Path(out_dir).iterdir())[-1]
        return last_run

    @staticmethod
    def _load_metrics(log_dir: Path) -> Dict[str, Any]:
        acc = EventAccumulator(str(log_dir))
        acc.Reload()
        psnr = [s.value for s in acc.Scalars("psnr/val")]
        steps = [s.step for s in acc.Scalars("psnr/val")]
        return {"psnr": psnr, "steps": steps}

    @staticmethod
    def _summarize_metrics(metrics: Dict[str, Any]) -> str:
        if not metrics["psnr"]:
            return "Training produced no PSNR values."
        best = max(metrics["psnr"])
        last = metrics["psnr"][-1]
        return f"Best PSNR: {best:.2f}. Last PSNR: {last:.2f}."

    def _query_llm(self, summary: str) -> Dict[str, Any]:
        if openai is None:
            raise ImportError("openai package is not installed")
        messages = [
            {"role": "system", "content": "You are a hyperparameter tuning assistant."},
            {"role": "user", "content": summary},
        ]
        response = openai.ChatCompletion.create(model=self.model, messages=messages)
        content = response["choices"][0]["message"]["content"]
        return json.loads(content)

    def _write_config(self) -> Path:
        """Write the current configuration to disk."""
        cfg = OmegaConf.merge(self.conf, self.overrides)
        cfg_path = self.out_dir / f"round_{self.current_round+1}.yaml"
        os.makedirs(self.out_dir, exist_ok=True)
        OmegaConf.save(cfg, cfg_path)
        return cfg_path

    def tune(self) -> None:
        """Run iterative tuning rounds."""
        for self.current_round in range(self.rounds):
            config = self._write_config()
            log_dir = self._run_training(config)
            metrics = self._load_metrics(log_dir)
            summary = self._summarize_metrics(metrics)
            params = self._query_llm(summary)
            self.overrides.update(params)


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM driven hyperparameter tuning")
    parser.add_argument("config", type=Path, help="Base config file")
    parser.add_argument("out_dir", type=Path, help="Output directory")
    parser.add_argument("--rounds", type=int, default=3, help="Number of tuning rounds")
    parser.add_argument("--model", type=str, default="gpt-4", help="OpenAI model name")
    parser.add_argument("--api-key", type=str, default=None, help="OpenAI API key (or set OPENAI_API_KEY env var)")
    parser.add_argument(
        "-o",
        "--override",
        action="append",
        default=[],
        help="Hydra style overrides to apply to the base config",
    )
    args = parser.parse_args()
    overrides: Dict[str, Any] = {}
    for ov in args.override:
        if "=" not in ov:
            continue
        k, v = ov.split("=", 1)
        try:
            overrides[k] = yaml.safe_load(v)
        except Exception:
            overrides[k] = v

    tuner = LLMAutoTuner(
        args.config,
        args.out_dir,
        args.rounds,
        args.model,
        args.api_key,
        overrides,
    )
    tuner.tune()


if __name__ == "__main__":
    main()
