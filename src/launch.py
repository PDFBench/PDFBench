import importlib
from typing import Type

from .configs import EvaluationArgs
from .utils import logging

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


def get_launch_args() -> EvaluationArgs:
    args = EvaluationArgs.parse()
    logging.set_global_logger()
    return args


def launch(config: EvaluationArgs) -> None:
    assert config.launch.metric_cls is not None

    module_path, class_name = EVALUATOR_IMPORTS[config.launch.metric_cls]
    module = importlib.import_module(module_path)
    evaluator_cls: Type = getattr(module, class_name)

    evaluator = evaluator_cls(config)
    evaluator.execute()


if __name__ == "__main__":
    launch(get_launch_args())
