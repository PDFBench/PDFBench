import json
import os
import shlex
import subprocess
from abc import ABC, abstractmethod
from copy import deepcopy

import accelerate
import torch
from torch.utils.data import DataLoader

from src.configs.parser import EvaluationArgs
from src.datasets import BaseDataset
from src.utils import logging


class EvaluationOutput:
    """
    Output for Evaluation Metric
    """

    def __init__(
        self,
        results: list[dict],
        metrics: list[str],
        design_batch_size: int,
    ) -> None:
        self._results = results
        self._metrics = metrics
        self._design_batch_size = design_batch_size

    @property
    def means(self):
        # TODO: Implement
        raise NotImplementedError

    @property
    def stds(self):
        # TODO: Implement
        raise NotImplementedError


class BaseMetric(ABC):
    def __init__(self, config: EvaluationArgs):
        self._config = config
        self._num_gpu = config.basic.num_gpu
        self._num_cpu = config.basic.num_cpu
        self._design_batch_size = config.basic.design_batch_size
        self._output_dir = config.basic.output_dir
        self._verbose = config.basic.verbose
        self._log_dir = config.basic.log_dir
        self._visualize = config.basic.visualize
        self._name: str
        self._speed_up: bool
        self.logger = logging.get_logger(
            f"{self.__class__.__module__}.{self.__class__.__name__}"
        )

    # region property
    @property
    def config(self) -> EvaluationArgs:
        return self._config

    @property
    def num_gpu(self) -> int:
        return self._num_gpu

    @property
    def design_batch_size(self) -> int:
        return self._design_batch_size

    @property
    def verbose(self) -> bool:
        return self._verbose

    @property
    def log_dir(self) -> str:
        return self._log_dir

    @property
    def output_dir(self) -> str:
        return self._output_dir

    @property
    def visualize(self) -> bool:
        return self._visualize

    @property
    def num_cpu(self) -> int:
        return self._num_cpu

    @property
    @abstractmethod
    def metrics(self) -> list[str]: ...

    @property
    def name(self) -> str:
        if self._name is None:
            raise ValueError("Output name of this Metric has not been set.")
        return self._name

    @property
    def output_path(self) -> str:
        return os.path.join(self.output_dir, f"{self.name}.json")

    @property
    def speed_up(self) -> bool:
        if self._speed_up is None:
            raise ValueError("Speed up of this Metric has not been set.")
        return self._speed_up

    # endregion

    def evaluate(self) -> EvaluationOutput:
        self.logger.info_rank0(f"Evaluating {self.name}")
        if os.path.exists(self.output_path):
            self.logger.info_rank0(f"Loading cache from {self.output_path}")
            with open(self.output_path, "r") as f:
                results = json.load(f)
        else:
            self.logger.info_rank0(
                f"Lauching evaluation subprocess for {self.name}"
            )

            # fmt: off
            cmd = []
            if self.speed_up:
                cmd.extend([
                    "accelerate", "launch",
                    "--multi_gpu",
                    "--num_processes", str(self.config.basic.num_gpu)
                ])
            else:
                cmd.append("python")

            cmd.extend([
                "-m", "src.launch",
                "--config_path", self.config.basic.config_path,
                "--lauch.metric_cls", self.__class__.__name__
            ])
            # fmt: on

            if self.speed_up:
                handler = (
                    "accelerate launch"
                    " --multi_gpu --num_processes {num_processes}"
                ).format(num_processes=self.config.basic.num_gpu)
            else:
                handler = "python"

            subprocess.run(
                args=shlex.split(
                    (
                        handler
                        + " -m src.launch --config_path {config_path} --launch.metric_cls {metric_cls}"
                    ).format(
                        config_path=self.config.basic.config_path,
                        metric_cls=self.__class__.__name__,
                    )
                ),
                env=deepcopy(os.environ),
                check=True,
            )
            with open(self.output_path, "r") as f:
                results = json.load(f)

        return EvaluationOutput(
            results=results,
            metrics=self.metrics,
            design_batch_size=self.design_batch_size,
        )


class MetricList:
    def __init__(self, metrics: list[BaseMetric], config: EvaluationArgs):
        self._metrics = metrics
        self._visualize = config.basic.visualize
        self._output_dir = config.basic.output_dir

    @property
    def metrics(self) -> list[BaseMetric]:
        return self._metrics

    @property
    def visualize(self) -> bool:
        return self._visualize

    @property
    def output_dir(self) -> str:
        return self._output_dir

    def evaluate(self):
        results: list[EvaluationOutput] = []
        for metric in self.metrics:
            results.append(metric.evaluate())
        return results


class BaseEvaluator(ABC):
    def __init__(self, config: EvaluationArgs) -> None:
        super().__init__()
        self._config = config
        self._num_gpu = config.basic.num_gpu
        self._num_cpu = config.basic.num_cpu
        self._design_batch_size = config.basic.design_batch_size
        self._output_dir = config.basic.output_dir
        self._name: str
        self.logger = logging.get_logger(
            f"{self.__class__.__module__}.{self.__class__.__name__}"
        )
        self._accelerator = accelerate.Accelerator()
        self._dataset: BaseDataset = config.basic.dataset_type.value(
            path=config.basic.input_path,
            design_batch_size=config.basic.design_batch_size,
        )
        self._dataloader: DataLoader

    @property
    def config(self) -> EvaluationArgs:
        return self._config

    @property
    def name(self) -> str:
        return self._name

    @property
    def output_dir(self) -> str:
        return self._output_dir

    @property
    def output_path(self) -> str:
        return os.path.join(self.output_dir, f"{self.name}.json")

    @property
    def design_batch_size(self) -> int:
        return self._design_batch_size

    @property
    def num_cpu(self) -> int:
        return self._num_cpu

    @property
    def num_gpu(self) -> int:
        return self._num_gpu

    @property
    def accelerator(self) -> accelerate.Accelerator:
        return self._accelerator

    @property
    def dataset(self) -> BaseDataset:
        return self._dataset

    @abstractmethod
    def execute(self) -> None: ...

    def __del__(self) -> None:
        torch.cuda.empty_cache()
        self.accelerator.end_training()
