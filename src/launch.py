import importlib
from typing import Type

from src.configs import EvaluationArgs

# from src.metrics import (
#     BaseEvaluator,
#     BertScoreEvaluator,
#     BertScoreMetric,
#     DiversityEvaluator,
#     DiversityMetric,
#     EvoLlamaScoreEvaluator,
#     EvoLlamaScoreMetric,
#     FoldabilityEvaluator,
#     FoldabilityMetric,
#     GOScoreEvaluator,
#     GOScoreMetric,
#     IdentityEvaluator,
#     IdentityMetric,
#     IPRScoreEvaluator,
#     IPRScoreMetric,
#     NoveltyEvaluator,
#     NoveltyMetric,
#     PerplexityEvaluator,
#     PerplexityMetric,
#     ProTrekScoreEvaluator,
#     ProTrekScoreMetric,
#     RepetitivenessEvaluator,
#     RepetitivenessMetric,
#     RetrievalAccuracyEvaluator,
#     RetrievalAccuracyMetric,
#     TMScoreEvaluator,
#     TMScoreMetric,
# )
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
# EVALUATOR_MAP: dict[str, Type[BaseEvaluator]] = {
#     BertScoreMetric.__name__: BertScoreEvaluator,
#     RepetitivenessMetric.__name__: RepetitivenessEvaluator,
#     PerplexityMetric.__name__: PerplexityEvaluator,
#     IdentityMetric.__name__: IdentityEvaluator,
#     FoldabilityMetric.__name__: FoldabilityEvaluator,
#     TMScoreMetric.__name__: TMScoreEvaluator,
#     ProTrekScoreMetric.__name__: ProTrekScoreEvaluator,
#     EvoLlamaScoreMetric.__name__: EvoLlamaScoreEvaluator,
#     RetrievalAccuracyMetric.__name__: RetrievalAccuracyEvaluator,
#     IPRScoreMetric.__name__: IPRScoreEvaluator,
#     GOScoreMetric.__name__: GOScoreEvaluator,
#     DiversityMetric.__name__: DiversityEvaluator,
#     NoveltyMetric.__name__: NoveltyEvaluator,
# }
logger = logging.get_logger(__name__)

EVALUATOR_IMPORTS: dict[str, tuple[str, str]] = {
    "BertScoreMetric": ("src.metrics", "BertScoreEvaluator"),
    "RepetitivenessMetric": ("src.metrics", "RepetitivenessEvaluator"),
    "PerplexityMetric": ("src.metrics", "PerplexityEvaluator"),
    "IdentityMetric": ("src.metrics", "IdentityEvaluator"),
    "FoldabilityMetric": ("src.metrics", "FoldabilityEvaluator"),
    "TMScoreMetric": ("src.metrics", "TMScoreEvaluator"),
    "ProTrekScoreMetric": ("src.metrics", "ProTrekScoreEvaluator"),
    "EvoLlamaScoreMetric": ("src.metrics", "EvoLlamaScoreEvaluator"),
    "RetrievalAccuracyMetric": ("src.metrics", "RetrievalAccuracyEvaluator"),
    "IPRScoreMetric": ("src.metrics", "IPRScoreEvaluator"),
    "GOScoreMetric": ("src.metrics.alignment.go_score", "GOScoreEvaluator"),
    "DiversityMetric": ("src.metrics", "DiversityEvaluator"),
    "NoveltyMetric": ("src.metrics", "NoveltyEvaluator"),
}


def get_lauch_args() -> EvaluationArgs:
    args = EvaluationArgs.parse()
    logging.set_global_logger()
    return args


# def launch(config: EvaluationArgs) -> None:
#     assert config.launch.metric_cls is not None
#     evaluator = EVALUATOR_MAP[config.launch.metric_cls](config)
#     evaluator.execute()


def launch(config: EvaluationArgs) -> None:
    assert config.launch.metric_cls is not None

    # 找到模块路径和类名
    module_path, class_name = EVALUATOR_IMPORTS[config.launch.metric_cls]

    # 动态 import 模块
    module = importlib.import_module(module_path)

    # 拿到类对象
    evaluator_cls: Type = getattr(module, class_name)

    evaluator = evaluator_cls(config)
    evaluator.execute()


if __name__ == "__main__":
    launch(get_lauch_args())
