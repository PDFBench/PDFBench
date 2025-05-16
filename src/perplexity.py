import json
import math
import multiprocessing as mp
import os
import warnings
from typing import Optional

import numpy as np
import torch
from tokenizers import Tokenizer
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
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


def compute_perplexity_progen(sequence: str, tokenizer, model) -> float:
    """
    Compute the perplexity of the protein sequence.
    :param sequence: sequence
    """
    sequence = sequence[: min(1024, len(sequence))]
    input_ids = torch.tensor(tokenizer.encode(sequence).ids).to(model.device)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)

    return math.exp(outputs.loss.item())


def compute_perplexity_protgpt2(sequence: str, tokenizer, model) -> float:
    """
    Compute the perplexity of the protein sequence.
    :param sequence: sequence
    """
    sequence = sequence[: min(1024, len(sequence))]
    sequence = "\n".join(
        [sequence[beg : beg + 60] for beg in range(0, len(sequence), 60)]
    )
    sequence = "<|endoftext|>\n" + sequence + "\n<|endoftext|>"

    input_ids = torch.tensor(
        tokenizer.encode(sequence, max_length=1024, truncation=True)
    ).unsqueeze(0)
    input_ids = input_ids.to(model.device)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    loss, logits = outputs[:2]
    return math.exp(loss)


def _main(uid: int, queue: mp.Queue, subset: list):
    results: list = [{"sequence": item["response"]} for item in subset]

    # region Perplexity besed on ProGPT2
    model_path = "/home/jhkuang/data/huggingface/hub/ProtGPT2"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path).to(f"cuda:{uid}")
    model.eval()

    idx = 0
    for item in tqdm(
        subset,
        desc=f"Process {uid} - Perplexity-ProGPT2",
        position=uid + 1,
        ncols=100,
    ):
        mutant = item["response"]

        try:
            res: dict = {
                "perplexity_protgpt2": compute_perplexity_protgpt2(
                    mutant, tokenizer, model
                ),
            }
            results[idx].update(res)
        except Exception as e:
            warnings.warn(f"Error in computing perplexity: \n{e}")
        idx += 1
    # endregion

    # region Perplexity besed on ProGen
    model = AutoModelForCausalLM.from_pretrained(
        "hugohrban/progen2-base", trust_remote_code=True
    ).to("cuda:0")
    model.eval()
    tokenizer = Tokenizer.from_pretrained("hugohrban/progen2-base")
    tokenizer.no_padding()

    idx = 0
    for item in tqdm(
        subset,
        desc=f"Process {uid} - Perplexity-ProGen",
        position=uid + 1,
        ncols=100,
    ):
        mutant = item["response"]
        try:
            res: dict = {
                "perplexity_progen": compute_perplexity_progen(
                    mutant, tokenizer, model
                ),
            }
            results[idx].update(res)
        except Exception as e:
            warnings.warn(f"Error in computing perplexity: \n{e}")
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
        "perplexity_protgpt2",
        "perplexity_progen",
    ]
    for metric in support_metrics:
        mean = np.mean([sample[metric] for sample in results])
        print(f"mean {metric}: {mean}")

    if save_plot:
        assert evaluation_dir
        for metric in support_metrics:
            boxplot(evaluation_dir, metric)


def test_compute_perplext_progen(seqs):
    model = AutoModelForCausalLM.from_pretrained(
        "hugohrban/progen2-base", trust_remote_code=True
    ).to("cuda:0")
    model.eval()
    tokenizer = Tokenizer.from_pretrained("hugohrban/progen2-base")
    tokenizer.no_padding()

    for seq in seqs:
        print(
            f"perplexity[ProGen] of {seq[:15]}: ",
            compute_perplexity_progen(seq, tokenizer, model),
        )


def test_comput_perplexity_protgpt2(seqs):
    model_path = "/home/jhkuang/data/huggingface/hub/ProtGPT2"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path).to("cuda:0")
    model.eval()

    for seq in seqs:
        print(
            f"perplexity[ProtGPT2] of {seq[:15]}: ",
            compute_perplexity_protgpt2(seq, tokenizer, model),
        )


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
    test_comput_perplexity_protgpt2([seq01, seq02, seq03])
    test_compute_perplext_progen([seq01, seq02, seq03])


if __name__ == "__main__":
    DEBUG = False
    if DEBUG:
        print("Debuging Perplexity")
        test()
    else:
        import fire

        fire.Fire(main)
