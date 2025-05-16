import json
import multiprocessing as mp
import os
import warnings
from typing import Optional

import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import (
    EsmModel,
    EsmTokenizer,
    logging,
)

from src.eval.utils import boxplot

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


def compute_bertscore(pred_seq: str, ref_seq: str, model, tokenizer) -> tuple:
    """
    compute BertScore
    :param pred_seq: sequence predicted by model
    :param ref_seq: sequence ground truth
    :param model: model used to calculate BertScore
    :param tokenizer: tokenizor used by model
    :return: bert_f1, bert_precision, bert_recall
    """

    def get_embeddings(sequence):
        tokens = tokenizer(
            sequence, return_tensors="pt", truncation=True, max_length=512
        ).to(model.device)
        with torch.no_grad():
            outputs = model(**tokens)
        embeddings = outputs.last_hidden_state[0][1:-1]

        return embeddings

    pred_embed = get_embeddings(pred_seq)
    ref_embed = get_embeddings(ref_seq)
    if pred_embed.size(0) == 0 or ref_embed.size(0) == 0:
        return 0.0, 0.0, 0.0

    similarity_matrix = torch.cosine_similarity(
        pred_embed.unsqueeze(1), ref_embed.unsqueeze(0), dim=-1
    )

    precision = similarity_matrix.max(dim=1)[0].mean().item()
    recall = similarity_matrix.max(dim=0)[0].mean().item()
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

    return f1, precision, recall


def _main(uid: int, queue: mp.Queue, subset: list):
    results: list = [dict() for _ in range(len(subset))]

    # region Bertscore based on ESM-2
    model_path = "/home/jhkuang/public/old_pretrain/esm2_t33_650M_UR50D"
    tokenizer = EsmTokenizer.from_pretrained(model_path)
    model = EsmModel.from_pretrained(model_path).to(f"cuda:{uid}")  # type: ignore
    model.eval()
    idx = 0
    for item in tqdm(
        subset,
        desc=f"Process {uid} - Bertscore",
        position=uid + 1,
        ncols=100,
    ):
        mutant = item["response"]
        wild_type = item["reference"]

        bert_f1, bert_precision, bert_recall = compute_bertscore(
            mutant, wild_type, model, tokenizer
        )
        res: dict = {
            "reference": wild_type,
            "response": mutant,
            "bertscore_f1": bert_f1,
            "bertscore_precision": bert_precision,
            "bertscore_recall": bert_recall,
        }
        results[idx].update(res)
        idx += 1
    # endregion

    queue.put(results)


def main(
    num_workers: int = 4,
    sequence_file: Optional[str] = None,
    evaluation_file: Optional[str] = None,
    evaluation_dir: Optional[str] = None,
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

            p = mp.Process(target=_main, args=(i, queue, subset))
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
        "bertscore_f1",
        "bertscore_precision",
        "bertscore_recall",
    ]
    for metric in support_metrics:
        mean = np.mean([sample[metric] for sample in results])
        print(f"mean {metric}: {mean:.2f}")

    if save_plot:
        assert evaluation_dir
        for metric in support_metrics:
            boxplot(evaluation_dir, metric)


def test():
    seq01 = (
        "IKVIDLMCPVVVVVVVVVVVLVTGALLGKGKGKGKGKGKATPKAVKKGKGKGKGKGKGKGKGK"
        "GKGKGKGKGKSGKGKGKGKGKGKGKGKGKGKGKGKGKGKGKSGKGKGKGKGKGKGKGKSGKGK"
        "GKGKGKGKGKGKSGKGKGKGKGKGKGKGKGKSGKGKGKGKGKGKGKGKGKSGKGKGKGKGKGK"
        "GKGKGKGKSGKGKGKGKGKGKGKGKGKGKGKSGKGKGKGKGKGKGKGKGKGKGKGKSGKGKGK"
        "GKGKGKGKGKSGKGKGKGKGKGKGKGKGKGKGKGKGKGKGKSGKGKGKGKGKGKGKGKSGKGK"
        "GKGKGKGKGKGKSGKGKGKGKGKGKGKGKGKSGKGKGKGKGKGKGKGKGKSGKGKGKGKGKGK"
        "GKGKGKGKSGKGKGKGKGKGKGKGKGKGKGKSGKGKGKGKGKGKGKGKGKGKGKGKSGKGKGK"
        "GKGKGKGKGKSGKGKGKGKGKGKGKGKGKGKGKGKGKGKGKSGKGKGKGKGKGKGKGKSGKGK"
        "GKGKGKGKGKGKSGKGKGKGKGKGKGKGKGKSGKGKGKGKGKGKGKGKGKSGKGKGKGKGKGK"
        "GKGKGKGKSGKGKGKGKGKGKGKGKGKGKGKSGKGKGKGKGKGKGKGKGKGKGKGKSGKGKGK"
        "GKGKGKGKGKGKGKGKGKGKSGKGKGKG"
    )
    seq02 = (
        "MQQDLAGKHILLGLTGGVACYKSAELCRLFIKAGATVQVVMTEAATQFITPVTMQALSGQPV"
        "YLSQWDARQANNMPHINLGREADAIVLAPASADFIARLVQGRSDELLSLLCLARPLQRVPLL"
        "VAPAMNREMWAHPATQRNLRQLADDGALVLGVGQGWQACGEAGDGRMLEPAELLEEVVAHFQ"
        "PKVLLGEHVVVTAGPTFEAMDPIRGITNHSSGKMGFAIARAAREAGARVTLVAGPVHLPTPR"
        "GVQRVDVASAQQMLQAVQAAVADASVFVATAAVADWRPADPQMHKIKKDGSGQTPVLRFVEN"
        "PDILHTVAQGERARGRQLFCVGFAAESENLLEHAKAKRLRKGIPLLVGNIGPLTFGQDDNSL"
        "LLVDEHGARELPRASKLALARELASEIAVRLRPWRG"
    )
    seq03 = (
        "IMLFCITMGGGKGKGKGKGKPVLKGKGKGKGKGKGIKGAVKGKGKGKGKGKGKGKGKGKGKGKGKGKGKG"
        "KGKGKGKGKGKGKGKGKGKGKGKGKGKEDVGKGKGKGKGKGKGKGKEDVGKGKGKGKGKGKGKGKGKGKG"
        "KGKGKGKGKGKGKEDVGKGKGKGKGKGKGKGKGKGKEDVGKGKGKGKGKGKGKGKGKGKGKEDVGKGKGK"
        "GKGKGKGKGKGKGKGKGKGKEDVGKGKGKGKGKGKGKGKGKGKGKGKGKGKGKGKGKGKGKGKGKGK"
    )
    model_path = "/home/jhkuang/public/old_pretrain/esm2_t33_650M_UR50D"
    tokenizer = EsmTokenizer.from_pretrained(model_path)
    model = EsmModel.from_pretrained(model_path).to("cuda:4")  # type: ignore
    model.eval()

    print(
        "perplexity: ",
        compute_bertscore(
            seq01,
            seq02,
            model,
            tokenizer,
        ),
    )
    print("perplexity: ", compute_bertscore(seq02, seq03, model, tokenizer))
    print("perplexity: ", compute_bertscore(seq03, seq01, model, tokenizer))


if __name__ == "__main__":
    DEBUG = False
    if DEBUG:
        print("Debuging Bert-Like Score")
        test()
    else:
        import fire

        fire.Fire(main)
