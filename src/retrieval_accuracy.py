import json
import multiprocessing as mp
import os
import random
import re
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import logging

from src.eval.ProTrek.model.ProTrek.protrek_trimodal_model import (
    ProTrekTrimodalModel,
)

logging.set_verbosity_error()
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="TypedStorage is deprecated.*",
)
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message="`clean_up_tokenization_spaces` was not set.*",
)


logging.set_verbosity_error()


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


def _main(
    uid: int,
    queue: mp.Queue,
    wholeset: list,
    subset: list,
    model_path: str,
    task: str,
    batch_size: int = 16,
):
    config = {
        "protein_config": os.path.join(model_path, "esm2_t33_650M_UR50D"),
        "text_config": os.path.join(
            model_path,
            "BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        ),
        "structure_config": os.path.join(model_path, "foldseek_t30_150M"),
        "load_protein_pretrained": False,
        "load_text_pretrained": False,
        "from_checkpoint": os.path.join(model_path, "ProTrek_650M_UniRef50.pt"),
    }
    model = ProTrekTrimodalModel(**config).eval().to(f"cuda:{uid}")

    # instruction to text
    get_text = get_molinst_text if task == "molinst" else get_cameo_text

    # pre calculate the embeddings
    seq_pool = [item["response"] for item in wholeset]
    with torch.no_grad():
        seq_embeds = []
        for idx in tqdm(range(0, len(seq_pool), batch_size), ncols=100):
            beg = idx
            end = min(idx + batch_size, len(seq_pool))
            seq_embeds.extend(model.get_protein_repr(seq_pool[beg:end]).cpu())
    seq_ref = {seq: embed for seq, embed in zip(seq_pool, seq_embeds)}
    text_pool = [get_text(item["instruction"]) for item in wholeset]
    with torch.no_grad():
        text_embeds = []
        for idx in tqdm(range(0, len(text_pool), batch_size), ncols=100):
            beg = idx
            end = min(idx + batch_size, len(text_pool))
            text_embeds.extend(model.get_text_repr(text_pool[beg:end]).cpu())
    text_ref = {text: embed for text, embed in zip(text_pool, text_embeds)}

    results: list = [dict() for _ in range(len(subset))]

    idx = 0
    for item in tqdm(
        subset,
        desc=f"Process {uid} - Retrieval Accuracy",
        position=uid + 1,
        ncols=100,
    ):
        reponse = item["response"]
        inst = get_text(item["instruction"])
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

    queue.put(results)


def main(
    sequence_file: str,
    evaluation_file: str,
    model_path: str,
    task: str,
    num_workers: int = 4,
    evaluation_dir: str = None,
    save_plot: bool = False,
):
    assert sequence_file and evaluation_file
    if not os.path.exists(evaluation_file):
        mp.set_start_method("spawn", force=True)

        with open(sequence_file, "r") as f:
            data: list = json.load(f)

        queue: mp.Queue = mp.Queue()
        processes: list = []
        for i in range(num_workers):
            piece = len(data) // num_workers
            beg_idx = i * piece
            end_idx = (i + 1) * piece if i != num_workers - 1 else len(data)
            subset = data[beg_idx:end_idx]

            p = mp.Process(
                target=_main,
                args=(
                    i,
                    queue,
                    data,
                    subset,
                    model_path,
                    task,
                ),
            )
            p.start()
            processes.append(p)

        results: list = [queue.get() for _ in range(len(processes))]
        results = [element for sublist in results for element in sublist]

        for p in processes:
            p.join()

        with open(evaluation_file, "w") as f:
            json.dump(results, f, indent=4)  # type: ignore
    else:
        print("Load processed evaluation file")
        with open(evaluation_file, "r") as f:
            results: list = json.load(f)

    support_metrics = [
        "retrieval_accuracy_4",
        "retrieval_accuracy_10",
        "retrieval_accuracy_20",
        "soft_retrieval_accuracy_4",
        "soft_retrieval_accuracy_10",
        "soft_retrieval_accuracy_20",
        "hard_retrieval_accuracy_4",
        "hard_retrieval_accuracy_10",
        "hard_retrieval_accuracy_20",
    ]
    for metric in support_metrics:
        mean = np.mean([sample[metric] for sample in results])
        print(f"mean {metric}: {mean:.4f}")


def test():
    device = 3
    print(device)
    sequence_poll_file = (
        "/home/jhkuang/data/cache/dynamsa/data/Molinst/inst2seq.json"
    )
    model_path = "/home/nwliu/data/pretrain/ProTrek_650M_UniRef50"
    config = {
        "protein_config": os.path.join(model_path, "esm2_t33_650M_UR50D"),
        "text_config": os.path.join(
            model_path,
            "BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        ),
        "structure_config": os.path.join(model_path, "foldseek_t30_150M"),
        "load_protein_pretrained": False,
        "load_text_pretrained": False,
        "from_checkpoint": os.path.join(model_path, "ProTrek_650M_UniRef50.pt"),
    }
    model = ProTrekTrimodalModel(**config).eval().to(f"cuda:{device}")

    sequence = "MVQSPMISCPLKQTNEIDWIQPLKDYIRQSYGEDPERYSQECATLNRLRQDMRGAGKDSATGRDLLYRYYGQLELLDLRFPVDENHIKISFTWYDAFTHKPTSQYSLAFEKASIIFNISAVLSCHAANQNRADDIGLKTAYHNFQASAGMFTYINENFLHAPSTDLNRETVKTLINITLAQGQEVFLEKQIMDHKKAGFLAKLASQASYLYAQAIEGTQEHAKGIFDKSWVTLLQVKSAHMGSVASYYQALADGESGSHGVAVARLQLAEKHSTSALSWAKSLPSSISPNTNLTSEAGPSLVDIVKFHLANVQSQLATFVKDNDFIYHQPVPSEAGLSAVSKLPAAKAIPVSELYQGQDIQRIIGPDIFQKLVPMSVTETASLYDEEKAKLIRAETEKVETADGEMAASLDYFKLPGSLNILKGGMDQEVMVDEEFQRWCQELAGHDSFAKAFDTLQDRKSEVLATLDQCAKQLDLEESVCEKMRSKYGADWSQQPSSRLNMTLRNDIRTYRDTVHEASASDAQLSATLRQYESDFDEMRSAGETDEADVLFQRAMIKAGSKQGKTKNGVTSPYSATEGSLLDDVYDDGVPSVAEQIARVESILKKLNLVKRERTQVLKDLKEKVRNDDISNVLILNKKSITGQESQLFEAELEKFHPHQMRIVQANHKQTALMKELTKTYGDLLQDKRVRAEQSKYESITRQRNSVMARYKKIYDSFNNLGSGIKQAQTFYAEMTETVDSLKKNVDTFINNRRSEGAQLLGQIEREK"
    instruction = "Construct a protein sequence with the desired structural and functional characteristics. 1. Target a Basic and acidic residues, Polar residues compositional bias in the protein's composition for improved properties.2. The protein must contain a signal peptide for proper functionality.3. The protein contains novel BRO1 domains that confer a unique function or activity.4. The protein design should be able to enhance the efficiency of protein transport. The designed protein sequence is "
    seq_embed = model.get_protein_repr([sequence]).cpu()
    text_embed = model.get_text_repr([instruction]).cpu()
    print(f"seq_embed: {seq_embed.shape}, text_embed: {text_embed.shape}")
    pos_score = F.cosine_similarity(text_embed, seq_embed).item()
    print(f"pos_score: {pos_score}")


if __name__ == "__main__":
    DEBUG = False
    if DEBUG:
        print("Debuging Retrieval Accuracy")
        test()
    else:
        import fire

        fire.Fire(main)
