import multiprocessing as mp
import os

import torch
from tqdm.auto import tqdm

from src.metrics import BaseEvaluator, BaseMetric
from src.metrics.alignment.ProTrek.model.ProTrek.protrek_trimodal_model import (
    ProTrekTrimodalModel,
)
from src.utils.multiprocess import multiprocess_evaluate


def protrek_score_evaluate_worker(
    queue: mp.Queue,
    pid: int,
    subset: list[dict],
    design_batch_size: int,
    protrek_path: str,
):
    if design_batch_size is None or protrek_path is None:
        raise ValueError(
            "Invalid kwargs: \n"
            f"design_batch_size: {design_batch_size}\n"
            f"pdb_cache_dir: {protrek_path}"
        )

    config = {
        "protein_config": os.path.join(protrek_path, "esm2_t33_650M_UR50D"),
        "text_config": os.path.join(
            protrek_path, "BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
        ),
        "structure_config": os.path.join(protrek_path, "foldseek_t30_150M"),
        "load_protein_pretrained": False,
        "load_text_pretrained": False,
        "from_checkpoint": os.path.join(
            protrek_path, "ProTrek_650M_UniRef50.pt"
        ),
    }
    model = ProTrekTrimodalModel(**config).eval().to(f"cuda:{pid}")  # type: ignore

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
                "reference": item["reference"],
                **{
                    f"response#{b}": item[f"response#{b}"]
                    for b in range(1, design_batch_size + 1)
                },
            }
            for b in range(1, design_batch_size + 1):
                res.update({f"response#{b}": item[f"response#{b}"]})

                text, sequence = item["instruction"], item[f"response#{b}"]
                text_embedding = model.get_text_repr([text])

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
            },
        )
        self.to_json(results)

    def execute(self) -> None:
        if self.speed_up:
            self._execute_acclerate()
        else:
            self._execute_manual_multiprocess()
