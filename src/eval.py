import os

from src.metrics import (
    BertScoreMetric,
    FoldabilityMetric,
    IdentityMetric,
    PerplexityMetric,
    RepetitivenessMetric,
)
from src.metrics.metric import EvaluationOutput

from .configs.parser import EvaluationArgs
from .metrics import MetricList
from .utils import logging
from .utils.visualization import to_csv

logger = logging.get_logger(__name__)


def get_eval_args() -> EvaluationArgs:
    args = EvaluationArgs.parse()
    logging.set_global_logger()
    logger.info_rank0(args.to_json())
    return args


def evaluate(config: EvaluationArgs) -> None:
    concerns = []
    if config.repeat.run:
        concerns.append(RepetitivenessMetric(config))
    if config.bert_score.run:
        concerns.append(BertScoreMetric(config))
    if config.perplexity.run:
        concerns.append(PerplexityMetric(config))
    if config.identity.run:
        concerns.append(IdentityMetric(config))
    if config.foldability.run:
        concerns.append(FoldabilityMetric(config))
    # TODO: remain metrics

    metrics: MetricList = MetricList(metrics=concerns, config=config)
    results: list[EvaluationOutput] = metrics.evaluate()

    if config.basic.visualize:
        to_csv(
            results=results,
            output_path=os.path.join(
                config.basic.output_dir, config.basic.visual_name
            ),
        )


def main():
    evaluate(get_eval_args())


if __name__ == "__main__":
    main()
