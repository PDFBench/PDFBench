import os

from src.metrics import (
    BertScoreMetric,
    DiversityMetric,
    EvoLlamaScoreMetric,
    FoldabilityMetric,
    GOScoreMetric,
    IdentityMetric,
    IPRScoreMetric,
    NoveltyMetric,
    PerplexityMetric,
    ProTrekScoreMetric,
    RepetitivenessMetric,
    RetrievalAccuracyMetric,
    TMScoreMetric,
)

from .configs import EvaluationArgs
from .metrics import MetricList
from .utils import logging
from .utils.visualization import to_json

logger = logging.get_logger(__name__)


def get_eval_args() -> EvaluationArgs:
    args = EvaluationArgs.parse()
    assert args.launch.metric_cls is None, "--lauch.metric_cls must be None"
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

    if config.foldability.run:
        concerns.append(FoldabilityMetric(config))
    if config.protrek_score.run:
        concerns.append(ProTrekScoreMetric(config))
    if config.retrievl_acc.run:
        concerns.append(RetrievalAccuracyMetric(config))
    if config.evollama_score.run:
        concerns.append(EvoLlamaScoreMetric(config))
    if config.novelty.run:
        concerns.append(NoveltyMetric(config))
    if config.go_score.run:
        concerns.append(GOScoreMetric(config))
    if config.identity.run:
        concerns.append(IdentityMetric(config))
    if config.tm_score.run:
        concerns.append(TMScoreMetric(config))
    if config.diversity.run:
        concerns.append(DiversityMetric(config))
    if config.ipr_score.run:
        concerns.append(IPRScoreMetric(config))

    metrics: MetricList = MetricList(metrics=concerns, config=config)
    results = metrics.evaluate()

    if config.basic.visualize:
        to_json(
            results=results,
            output_path=os.path.join(
                config.basic.output_dir, config.basic.visual_name + ".json"
            ),
        )


def main():
    evaluate(get_eval_args())


if __name__ == "__main__":
    main()
