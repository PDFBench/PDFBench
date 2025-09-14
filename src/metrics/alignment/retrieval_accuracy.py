import multiprocessing as mp
import random
import re

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from src.configs.alignment_args import RetrievalDifficulty
from src.datasets.dataset import BaseDataset
from src.utils.context_manager import suppress_all_output
from src.utils.multiprocess import multiprocess_evaluate

from ..metric import BaseEvaluator, BaseMetric

with suppress_all_output():
    from .ProTrek.model.ProTrek.protrek_trimodal_model import (
        ProTrekTrimodalModel,
    )
    from .utils import load_protrek


def get_cameo_text(instruction: str) -> str:
    # We only keep the keyword part of the instruction for evaluation
    keyword = instruction.removesuffix("The designed protein sequence is ")
    keyword = re.search(r":\s*(.*)", keyword[:-2]).group(1)  # type: ignore
    return keyword.strip()


def get_molinst_text(instruction: str) -> str:
    # We only keep the function part of the instruction for evaluation
    function = re.sub(r"^.*?(1\.)", r"\1", instruction)
    function = function.removesuffix("The designed protein sequence is ")
    return function.strip()


def topk_by_similarity(query_vector, vector_set, topk=5, most_similar=True):
    vector_set = torch.stack(vector_set).to(query_vector.device)
    query_vector = query_vector.reshape(1, -1)
    sims = F.cosine_similarity(query_vector, vector_set)
    if most_similar:
        indices = torch.topk(sims, k=topk, largest=True).indices
    else:
        indices = torch.topk(sims, k=topk, largest=False).indices
    return indices


def compute_retrieval_accuracy_batch(
    text_embed,
    seq_embed,
    neg_seq_embeds,
) -> tuple[int, int, int]:
    # access retrieval accuracy for 4 10 20
    pos_score = F.cosine_similarity(
        text_embed.unsqueeze(0), seq_embed.unsqueeze(0)
    ).item()
    scores_4 = [pos_score]
    scores_10 = [pos_score]
    scores_20 = [pos_score]

    for neg_embed in random.sample(neg_seq_embeds, 4 - 1):
        scores_4.append(
            F.cosine_similarity(text_embed, neg_embed.unsqueeze(0)).item()
        )
    for neg_embed in random.sample(neg_seq_embeds, 10 - 1):
        scores_10.append(
            F.cosine_similarity(text_embed, neg_embed.unsqueeze(0)).item()
        )
    for neg_embed in random.sample(neg_seq_embeds, 20 - 1):
        scores_20.append(
            F.cosine_similarity(text_embed, neg_embed.unsqueeze(0)).item()
        )

    return (
        1 if pos_score == max(scores_4) else 0,
        1 if pos_score == max(scores_10) else 0,
        1 if pos_score == max(scores_20) else 0,
    )


def compute_retrieval_accuracy_batch_soft_hard(
    text_embed,
    seq_embed,
    neg_seq_embeds,
    text_embeds,
    soft: bool = True,
) -> tuple[int, int, int]:
    pos_score = F.cosine_similarity(
        text_embed.unsqueeze(0), seq_embed.unsqueeze(0)
    ).item()
    scores_4 = [pos_score]
    scores_10 = [pos_score]
    scores_20 = [pos_score]

    # The first index is the text_embed itself (hard) or redundant one (soft), so discard it.
    text_indices = topk_by_similarity(
        text_embed, text_embeds, topk=20, most_similar=not soft
    )[1:]

    for idx in text_indices:
        neg_embed = neg_seq_embeds[idx]

        if len(scores_4) < 4:
            scores_4.append(
                F.cosine_similarity(text_embed, neg_embed.unsqueeze(0)).item()
            )
        if len(scores_10) < 10:
            scores_10.append(
                F.cosine_similarity(text_embed, neg_embed.unsqueeze(0)).item()
            )
        if len(scores_20) < 20:
            scores_20.append(
                F.cosine_similarity(text_embed, neg_embed.unsqueeze(0)).item()
            )

    return (
        1 if pos_score == max(scores_4) else 0,
        1 if pos_score == max(scores_10) else 0,
        1 if pos_score == max(scores_20) else 0,
    )


def compute_retrieval_accuracy(
    model: ProTrekTrimodalModel,
    inst: str,
    seq: str,
    neg_seq_pool: list[str],
    num_neg: int,
) -> int:
    neg_seqs = random.sample(neg_seq_pool, num_neg)

    # access retrieval accuracy
    text_embed = model.get_text_repr([inst]).cpu()
    pos_embed = model.get_protein_repr([seq]).cpu()
    scores = [F.cosine_similarity(text_embed, pos_embed).item()]
    for neg in neg_seqs:
        neg_embed = model.get_protein_repr([neg])

        scores.append(F.cosine_similarity(text_embed, neg_embed).item())

    torch.cuda.empty_cache()
    return 1 if scores[0] == max(scores) else 0


def evollama_score_evaluate_worker(
    queue: mp.Queue,
    pid: int,
    subset: list[dict],
    wholeset: BaseDataset,
    design_batch_size: int,
    protrek_path: str,
    protrek_batch_size: int,
    retrieval_difficulties: list[str],
):
    if (
        design_batch_size is None
        or protrek_path is None
        or protrek_batch_size is None
        or retrieval_difficulties is None
    ):
        raise ValueError(
            "Invalid kwargs: \n"
            f"design_batch_size: {design_batch_size}\n"
            f"protrek_path: {protrek_path}\n"
            f"protrek_batch_size: {protrek_batch_size}\n"
            f"design_batch_size: {design_batch_size}\n"
            f"retrieval_difficulties: {retrieval_difficulties}"
        )

    model: ProTrekTrimodalModel = load_protrek(protrek_path, pid)

    # pre-calculate the embeddings
    # sequence embedding pool
    seq_pool = [
        item[f"response#{b}"][:2048]  # type: ignore
        for item in wholeset
        for b in range(1, design_batch_size + 1)
    ]
    with torch.no_grad():
        seq_embeds = []
        for idx in tqdm(
            range(0, len(seq_pool), protrek_batch_size),
            desc="ProTrek-Sequence-Embedding",
            ncols=100,
            disable=pid != 0,
            postfix=f"Batch Size: {protrek_batch_size}",
        ):
            beg = idx
            end = min(idx + protrek_batch_size, len(seq_pool))
            seq_embeds.extend(model.get_protein_repr(seq_pool[beg:end]).cpu())
            torch.cuda.empty_cache()
    seq_ref = {seq: embed for seq, embed in zip(seq_pool, seq_embeds)}
    # text embedding pool
    text_pool = [wholeset.function(item["instruction"]) for item in wholeset]  # type: ignore
    with torch.no_grad():
        text_embeds = []
        for idx in tqdm(
            range(
                0,
                len(text_pool),
                protrek_batch_size,
            ),
            ncols=100,
            desc="ProTrek-Text-Embedding",
            disable=pid != 0,
            postfix=f"Batch Size: {protrek_batch_size}",
        ):
            beg = idx
            end = min(idx + protrek_batch_size, len(text_pool))
            text_embeds.extend(model.get_text_repr(text_pool[beg:end]).cpu())
            torch.cuda.empty_cache()
    text_ref = {text: embed for text, embed in zip(text_pool, text_embeds)}

    results: list = [dict() for _ in range(len(subset))]
    for idx, item in enumerate(
        tqdm(
            subset,
            desc="Retrieval Accuracy",
            ncols=100,
            disable=pid != 0,
        )
    ):
        res = {
            "instruction": item["instruction"],
            "function": wholeset.function(item["instruction"]),
            "reference": item["reference"],
            **{
                f"response#{b}": item[f"response#{b}"]
                for b in range(1, design_batch_size + 1)
            },
        }

        for b in range(1, design_batch_size + 1):
            res.update({f"response#{b}": item[f"response#{b}"]})

            pos_seq_embed = seq_ref[item[f"response#{b}"][:2048]]
            pos_text_embed = text_ref[wholeset.function(item["instruction"])]

            if RetrievalDifficulty.Hard.name in retrieval_difficulties:
                rAc4, rAc10, rAc20 = compute_retrieval_accuracy_batch_soft_hard(
                    pos_text_embed,
                    pos_seq_embed,
                    seq_embeds,
                    text_embeds,
                    soft=False,
                )
                res.update(
                    {
                        f"RetrievalAccuracy[4]-Hard#{b}": rAc4,
                        f"RetrievalAccuracy[10]-Hard#{b}": rAc10,
                        f"RetrievalAccuracy[20]-Hard#{b}": rAc20,
                    }
                )
            if RetrievalDifficulty.Soft.name in retrieval_difficulties:
                rAc4, rAc10, rAc20 = compute_retrieval_accuracy_batch_soft_hard(
                    pos_text_embed,
                    pos_seq_embed,
                    seq_embeds,
                    text_embeds,
                    soft=True,
                )
                res.update(
                    {
                        f"RetrievalAccuracy[4]-Soft#{b}": rAc4,
                        f"RetrievalAccuracy[10]-Soft#{b}": rAc10,
                        f"RetrievalAccuracy[20]-Soft#{b}": rAc20,
                    }
                )
            if RetrievalDifficulty.Normal.name in retrieval_difficulties:
                rAc4, rAc10, rAc20 = compute_retrieval_accuracy_batch(
                    pos_text_embed, pos_seq_embed, seq_embeds
                )
                res.update(
                    {
                        f"RetrievalAccuracy[4]-Normal#{b}": rAc4,
                        f"RetrievalAccuracy[10]-Normal#{b}": rAc10,
                        f"RetrievalAccuracy[20]-Normal#{b}": rAc20,
                    }
                )

        results[idx].update(res)

    queue.put((pid, results))


class RetrievalAccuracyMetric(BaseMetric):
    def __init__(self, config):
        super().__init__(config)
        self._name = config.retrievl_acc.name
        self.retrieval_difficulties = config.retrievl_acc.retrieval_difficulties

    @property
    def metrics(self) -> list[str]:
        _metrics = []
        for difficulty in RetrievalDifficulty:
            if difficulty.name not in self.retrieval_difficulties:
                _metrics.extend(
                    [
                        f"RetrievalAccuracy[4]-{difficulty.name}",
                        f"RetrievalAccuracy[10]-{difficulty.name}",
                        f"RetrievalAccuracy[20]-{difficulty.name}",
                    ]
                )
        return _metrics

    def summary(self, results) -> dict[str, float]:
        _summary = {}
        bs = self.design_batch_size
        if bs == 1:
            for difficulty in RetrievalDifficulty:
                if difficulty.name in self.retrieval_difficulties:
                    _summary.update(
                        {
                            f"RetrievalAccuracy[4]-{difficulty.name}": results[
                                f"RetrievalAccuracy[4]-{difficulty.name}#1"
                            ].mean()
                            * 100,
                            f"RetrievalAccuracy[10]-{difficulty.name}": results[
                                f"RetrievalAccuracy[10]-{difficulty.name}#1"
                            ].mean()
                            * 100,
                            f"RetrievalAccuracy[20]-{difficulty.name}": results[
                                f"RetrievalAccuracy[20]-{difficulty.name}#1"
                            ].mean()
                            * 100,
                        }
                    )
        else:
            rAccs = {
                f"{difficulty.name}": [
                    results[
                        f"RetrievalAccuracy[{total}]-{difficulty.name}#{b}"
                    ].mean()
                    * 100
                    for total in [4, 10, 20]
                    for b in range(1, bs + 1)
                ]
                for difficulty in RetrievalDifficulty
                if difficulty.name in self.retrieval_difficulties
            }

            for difficulty in RetrievalDifficulty:
                if difficulty.name in self.retrieval_difficulties:
                    for idx, tot in enumerate([4, 10, 20]):
                        _summary[
                            f"RetrievalAccuracy[{tot}]-{difficulty.name}"
                        ] = np.mean(
                            rAccs[f"{difficulty.name}"][
                                idx * bs : (idx + 1) * bs
                            ]
                        )
                        _summary.update(
                            {
                                f"RetrievalAccuracy[{tot}]-{difficulty.name}"
                                f"#{(b - 1) % b + 1}": rAccs[
                                    f"{difficulty.name}"
                                ][b - 1]
                                for b in range(idx * bs + 1, (idx + 1) * bs + 1)
                            }
                        )

        return _summary


class RetrievalAccuracyEvaluator(BaseEvaluator):
    def __init__(self, config):
        super().__init__(config)
        self._name = config.retrievl_acc.name
        self.protrek_path = config.retrievl_acc.protrek_path
        self.protrek_batch_size = config.retrievl_acc.protrek_batch_size
        self.retrieval_difficulties = config.retrievl_acc.retrieval_difficulties

    def _execute_acclerate(self) -> None:
        raise NotImplementedError

    def _execute_manual_multiprocess(self) -> None:
        results = multiprocess_evaluate(
            dataset=self.dataset,
            eval_worker=evollama_score_evaluate_worker,
            num_workers=self.num_gpu,
            kwargs={
                "wholeset": self.dataset,
                "design_batch_size": self.design_batch_size,
                "protrek_path": self.protrek_path,
                "protrek_batch_size": self.protrek_batch_size,
                "retrieval_difficulties": self.retrieval_difficulties,
            },
        )
        self.to_json(results)

    def execute(self) -> None:
        if self.speed_up:
            self._execute_acclerate()
        else:
            self._execute_manual_multiprocess()
