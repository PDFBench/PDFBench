import json
import os
import subprocess
from abc import ABC, abstractmethod
from copy import deepcopy

import accelerate

from src.configs.parser import EvaluationArgs
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
        self.logger = logging.get_logger(
            f"{self.__class__.__module__}.{self.__class__.__name__}"
        )

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
            self.logger.info_rank0(
                (
                    "accelerate launch --multi_gpu --num_processes "
                    "{num_processes} -m src.launch --config_path {config_path}"
                    " --launch.metric_cls {metric_cls}"
                ).format(
                    num_processes=self.config.basic.num_gpu,
                    config_path=self.config.basic.config_path,
                    metric_cls=self.__class__.__name__,
                )
            )
            subprocess.run(
                args=(
                    "accelerate launch --multi_gpu --num_processes "
                    "{num_processes} -m src.lauch --config_path {config_path}"
                    " --lauch.metric_cls {metric_cls}"
                )
                .format(
                    num_processes=self.config.basic.num_gpu,
                    config_path=self.config.basic.config_path,
                    metric_cls=self.__class__.__name__,
                )
                .split(),
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
        self._accelerator = accelerate.Accelerator()
        self._data

    @property
    def config(self) -> EvaluationArgs:
        return self._config

    @property
    def accelerator(self) -> accelerate.Accelerator:
        return self._accelerator

    @abstractmethod
    def execute(self) -> None: ...
