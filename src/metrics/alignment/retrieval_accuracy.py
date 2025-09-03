import multiprocessing as mp
import random
import re

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from src.configs.alignment_args import RetrievalDifficulty
from src.datasets.dataset import BaseDataset
from src.metrics import BaseEvaluator, BaseMetric
from src.metrics.alignment.ProTrek.model.ProTrek.protrek_trimodal_model import (
    ProTrekTrimodalModel,
)
from src.metrics.alignment.utils import load_protrek
from src.utils.multiprocess import multiprocess_evaluate


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
) -> dict:
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

    return {
        "retrieval_accuracy_4": 1 if pos_score == max(scores_4) else 0,
        "retrieval_accuracy_10": 1 if pos_score == max(scores_10) else 0,
        "retrieval_accuracy_20": 1 if pos_score == max(scores_20) else 0,
    }


def compute_retrieval_accuracy_batch_soft_hard(
    text_embed,
    seq_embed,
    neg_seq_embeds,
    text_embeds,
    soft: bool = True,
):
    pos_score = F.cosine_similarity(
        text_embed.unsqueeze(0), seq_embed.unsqueeze(0)
    ).item()
    scores_4 = [pos_score]
    scores_10 = [pos_score]
    scores_20 = [pos_score]

    text_indices = topk_by_similarity(
        text_embed, text_embeds, topk=20, most_similar=not soft
    )[
        1:
    ]  # The first index is the text_embed itself (hard) or redundant one (soft), so discard it.

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

    prefix = "soft" if soft else "hard"

    return {
        f"{prefix}_retrieval_accuracy_4": 1
        if pos_score == max(scores_4)
        else 0,
        f"{prefix}_retrieval_accuracy_10": 1
        if pos_score == max(scores_10)
        else 0,
        f"{prefix}_retrieval_accuracy_20": 1
        if pos_score == max(scores_20)
        else 0,
    }


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
    wholeset: BaseDataset,
    subset: list[dict],
    design_batch_size: int,
    protrek_path: str,
    protrek_batch_size: int,
):
    if design_batch_size is None or protrek_path is None:
        raise ValueError(
            "Invalid kwargs: \n"
            f"design_batch_size: {design_batch_size}\n"
            f"protrek_path: {protrek_path}\n"
            f"protrek_batch_size: {protrek_batch_size}\n"
            f"design_batch_size: {design_batch_size}\n"
            f"protrek_path: {protrek_path}"
        )

    model: ProTrekTrimodalModel = load_protrek(protrek_path, pid)

    # pre calculate the embeddings
    seq_pool = [item["response"] for item in wholeset]  # type: ignore
    with torch.no_grad():
        seq_embeds = []
        for idx in tqdm(range(0, len(seq_pool), protrek_batch_size), ncols=100):
            beg = idx
            end = min(idx + protrek_batch_size, len(seq_pool))
            seq_embeds.extend(model.get_protein_repr(seq_pool[beg:end]).cpu())
    seq_ref = {seq: embed for seq, embed in zip(seq_pool, seq_embeds)}
    text_pool = [wholeset.function(item["instruction"]) for item in wholeset]  # type: ignore
    with torch.no_grad():
        text_embeds = []
        for idx in tqdm(
            range(0, len(text_pool), protrek_batch_size), ncols=100
        ):
            beg = idx
            end = min(idx + protrek_batch_size, len(text_pool))
            text_embeds.extend(model.get_text_repr(text_pool[beg:end]).cpu())
    text_ref = {text: embed for text, embed in zip(text_pool, text_embeds)}

    results: list = [dict() for _ in range(len(subset))]

    idx = 0
    for item in tqdm(
        subset,
        desc="Retrieval Accuracy",
        position=pid + 1,
        ncols=100,
        disable=pid != 0,
    ):
        reponse = item["response"]
        inst = wholeset.function(item["instruction"])
        res: dict = {
            "response": reponse,
            "instruction": inst,
        }

        pos_seq_embed = seq_ref[reponse]
        pos_text_embed = text_ref[inst]
        hard = compute_retrieval_accuracy_batch_soft_hard(
            pos_text_embed, pos_seq_embed, seq_embeds, text_embeds, soft=False
        )
        soft = compute_retrieval_accuracy_batch_soft_hard(
            pos_text_embed, pos_seq_embed, seq_embeds, text_embeds, soft=True
        )
        res.update(
            compute_retrieval_accuracy_batch(
                pos_text_embed, pos_seq_embed, seq_embeds
            )
        )
        res.update(hard)
        res.update(soft)
        results[idx].update(res)
        idx += 1

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
                _metrics.append(f"RetrievalAccuracy-{difficulty.name}")
        return _metrics


class RetrievalAccuracyEvaluator(BaseEvaluator):
    def __init__(self, config):
        super().__init__(config)
        self._name = config.retrievl_acc.name
        self.protrek_path = config.retrievl_acc.protrek_path
        self.protrek_batch_size = config.retrievl_acc.protrek_batch_size
        self.retrieval_difficulties = config.retrievl_acc.retrieval_difficulties

        self.desc_pool = config.retrievl_acc.desc_pool
        self.interpro_pool = config.retrievl_acc.ipr_pool
        self.go_pool = config.retrievl_acc.go_pool
        self.ec_pool = config.retrievl_acc.ec_pool

    def _execute_acclerate(self) -> None:
        raise NotImplementedError

    def _execute_manual_multiprocess(self) -> None:
        results = multiprocess_evaluate(
            dataset=self.dataset,
            eval_worker=evollama_score_evaluate_worker,
            num_workers=self.num_gpu,
            kwargs={
                "design_batch_size": self.design_batch_size,
                "protrek_path": self.protrek_path,
                "protrek_batch_size": self.protrek_batch_size,
                "retrieval_difficulties": self.retrieval_difficulties,
                "molinst_pool": self.desc_pool,
                "interpro_pool": self.interpro_pool,
                "go_pool": self.go_pool,
                "ec_pool": self.ec_pool,
            },
        )
        self.to_json(results)

    def execute(self) -> None:
        if self.speed_up:
            self._execute_acclerate()
        else:
            self._execute_manual_multiprocess()
