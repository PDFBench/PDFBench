from abc import ABC, abstractmethod

from src.configs.parser import EvaluationArgs
from src.datasets import BaseDataset


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
        self._num_gpu = config.basic.num_gpu
        self._num_cpu = config.basic.num_cpu
        self._design_batch_size = config.basic.design_batch_size
        self._output_dir = config.basic.output_dir
        self._verbose = config.basic.verbose
        self._log_dir = config.basic.log_dir
        self._visualize = config.basic.visualize

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

    @abstractmethod
    def evaluate(self, dataset: BaseDataset, **kwargs) -> EvaluationOutput: ...


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

    def evaluate(self, dataset: BaseDataset):
        results: list[EvaluationOutput] = []
        for metric in self.metrics:
            results.append(metric.evaluate(dataset=dataset))
        return results
