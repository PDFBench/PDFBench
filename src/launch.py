from typing import Type

from src.configs.parser import EvaluationArgs
from src.metrics import (
    BaseEvaluator,
    BertScoreEvaluator,
)

from .utils import logging

logger = logging.get_logger(__name__)

EVALUATOR_MAP: dict[str, Type[BaseEvaluator]] = {  # FIXME: Str 2 Type
    "BertScoreMetric": BertScoreEvaluator
}


def get_lauch_args() -> EvaluationArgs:
    args = EvaluationArgs.parse()
    logging.set_global_logger()
    return args


def launch(config: EvaluationArgs) -> None:
    assert config.launch.metric_cls is not None
    evaluator = EVALUATOR_MAP[config.launch.metric_cls](config)
    evaluator.execute()


if __name__ == "__main__":
    launch(get_lauch_args())
