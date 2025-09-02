from typing import Type

from src.configs import EvaluationArgs
from src.metrics import BaseEvaluator, BertScoreEvaluator, BertScoreMetric

from .utils import logging

logger = logging.get_logger(__name__)

# EVALUATOR_MAP: dict[MetricType, Type[BaseEvaluator]] = {
#     MetricType.BertScoreMetric: BertScoreEvaluator,
#     MetricType.RepetitivenessMetric: BertScoreEvaluator,
#     MetricType.PerplexityMetric: BertScoreEvaluator,
#     MetricType.IdentityMetric: BertScoreEvaluator,
#     MetricType.FoldabilityMetric: BertScoreEvaluator,
#     # MetricType.TMScoreMetric: TMScoreEvaluator,
#     # MetricType.ProTrekScoreMetric: ProTrekScoreEvaluator,
#     # MetricType.EvoLlamaScoreMetric: EvoLlamaScoreEvaluator,
#     # MetricType.RetrievalAccuracyMetric: RetrievalAccuracyEvaluator,
#     # MetricType.KeywordRecoveryMetric: KeywordRecoveryEvaluator,
#     # MetricType.DiversityMetric: DiversityEvaluator,
#     # MetricType.NoveltyMetric: NoveltyEvaluator,
# }
EVALUATOR_MAP: dict[str, Type[BaseEvaluator]] = {
    BertScoreMetric.__name__: BertScoreEvaluator,
    # "RepetitivenessMetric": BertScoreEvaluator,
    # "PerplexityMetric": BertScoreEvaluator,
    # "IdentityMetric": BertScoreEvaluator,
    # "FoldabilityMetric": BertScoreEvaluator,
    # MetricType.TMScoreMetric: TMScoreEvaluator,
    # MetricType.ProTrekScoreMetric: ProTrekScoreEvaluator,
    # MetricType.EvoLlamaScoreMetric: EvoLlamaScoreEvaluator,
    # MetricType.RetrievalAccuracyMetric: RetrievalAccuracyEvaluator,
    # MetricType.KeywordRecoveryMetric: KeywordRecoveryEvaluator,
    # MetricType.DiversityMetric: DiversityEvaluator,
    # MetricType.NoveltyMetric: NoveltyEvaluator,
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
