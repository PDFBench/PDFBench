import os

from src.datasets.dataset import BaseDataset
from src.metrics.sequence.repetitiveness import RepetitivenessMetric

from .configs.parser import EvaluationArgs
from .metrics import MetricList
from .utils import logging
from .utils.visualization import to_csv

logger = logging.get_logger(__name__)


def get_eval_args() -> EvaluationArgs:
    args = EvaluationArgs.parse()
    args.init()
    logging.set_global_logger()
    logger.info_rank0(args.to_json())
    return args


def evaluate(config: EvaluationArgs) -> None:
    concerns = []
    if config.repeat.run:
        concerns.append(RepetitivenessMetric(config))
    # TODO: remain metrics

    metrics: MetricList = MetricList(metrics=concerns, config=config)
    dataset: BaseDataset = config.basic.dataset_type.value(
        path=config.basic.input_path,
        design_batch_size=config.basic.design_batch_size,
    )
    results = metrics.evaluate(dataset=dataset)

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
