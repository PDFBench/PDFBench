from abc import ABC, abstractmethod

from ..datasets import BaseDataset


class EvaluationOutput:
    """
    Output for Evaluation Metric
    """

    _results: list[dict]
    _metrics: list[str]

    def __init__(
        self,
        results: list[dict],
        metrics: list[str],
        batch_size: int,
    ) -> None:
        self._results = results
        self._metrics = metrics

    @property
    def means(self):
        # TODO: Implement
        raise NotImplementedError

    @property
    def stds(self):
        # TODO: Implement
        raise NotImplementedError


class BaseMetric(ABC):
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def evaluate(self, dataset: BaseDataset, **kwargs) -> EvaluationOutput: ...
