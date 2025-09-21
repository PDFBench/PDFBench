import multiprocessing as mp
from typing import Callable

import numpy as np
import torch
from tqdm.auto import tqdm

from src.utils.context_manager import suppress_all_output

with suppress_all_output():
    from src.metrics.alignment.utils import load_protrek
    from src.utils.multiprocess import multiprocess_evaluate

from ..metric import BaseEvaluator, BaseMetric


def protrek_score_evaluate_worker(
    queue: mp.Queue,
    pid: int,
    subset: list[dict],
    design_batch_size: int,
    protrek_path: str,
    functional: Callable,
):
    if design_batch_size is None or protrek_path is None:
        raise ValueError(
            "Invalid kwargs: \n"
            f"design_batch_size: {design_batch_size}\n"
            f"pdb_cache_dir: {protrek_path}"
        )

    model = load_protrek(protrek_path, pid)

    results = []
    with torch.no_grad():
        for item in tqdm(
            subset,
            desc="Protrek Score",
            ncols=100,
            disable=pid != 0,
        ):
            res = {
                "instruction": item["instruction"],
                "function": functional(item["instruction"]),
                "reference": item["reference"],
                **{
                    f"response#{b}": item[f"response#{b}"]
                    for b in range(1, design_batch_size + 1)
                },
            }
            for b in range(1, design_batch_size + 1):
                res.update({f"response#{b}": item[f"response#{b}"]})

                function, sequence = (
                    functional(item["instruction"]),
                    item[f"response#{b}"],
                )
                text_embedding = model.get_text_repr([function])

                sequence_embedding = model.get_protein_repr([sequence])
                protrek_score = torch.nn.functional.cosine_similarity(
                    sequence_embedding, text_embedding
                ).item()

                res.update({f"ProTrekScore#{b}": protrek_score})

            results.append(res)

    queue.put((pid, results))


class ProTrekScoreMetric(BaseMetric):
    def __init__(self, config):
        super().__init__(config)
        self._name = config.protrek_score.name
        self.protrek_path = config.protrek_score.protrek_path

    @property
    def metrics(self) -> list[str]:
        return ["ProTrekScore"]

    def summary(self, results) -> dict[str, float]:
        bs = self.design_batch_size
        _summary = {}
        if bs == 1:
            _summary["ProTrekScore"] = results["ProTrekScore#1"].mean() * 100
        else:
            protrek_scores = [
                results[f"ProTrekScore#{b}"].mean() * 100
                for b in range(1, bs + 1)
            ]
            _summary["ProTrekScore"] = (
                rf"{np.mean(protrek_scores):.2f}"
                r"\(\pm\)"
                rf"{np.std(protrek_scores, ddof=1):.2f}"
            )
            _summary.update(
                {
                    f"ProTrekScore#{b}": protrek_scores[b - 1]
                    for b in range(1, bs + 1)
                }
            )
        return _summary


class ProTrekScoreEvaluator(BaseEvaluator):
    def __init__(self, config):
        super().__init__(config)
        self._name = config.protrek_score.name
        self.protrek_path = config.protrek_score.protrek_path

    def _execute_acclerate(self) -> None:
        raise NotImplementedError

    def _execute_manual_multiprocess(self) -> None:
        results = multiprocess_evaluate(
            dataset=self.dataset,
            eval_worker=protrek_score_evaluate_worker,
            num_workers=self.num_gpu,
            kwargs={
                "design_batch_size": self.design_batch_size,
                "protrek_path": self.protrek_path,
                "functional": self.dataset.function,
            },
        )
        self.to_json(results)

    def execute(self) -> None:
        if self.speed_up:
            self._execute_acclerate()
        else:
            self._execute_manual_multiprocess()
