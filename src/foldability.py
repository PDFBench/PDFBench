import hashlib
import json
import multiprocessing as mp
import os
import warnings
from typing import Dict, List

import biotite.structure.io as bsio
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import EsmForProteinFolding, EsmTokenizer, logging

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


def get_md5_sequence(sequence: str) -> str:
    return hashlib.md5(sequence.encode()).hexdigest()


def get_pae(output):
    pae = (
        output["aligned_confidence_probs"][0].cpu().numpy() * np.arange(64)
    ).mean(-1) * 31
    mask = output["atom37_atom_exists"][0, :, 1] == 1
    mask = mask.cpu()
    pae = pae[mask, :][:, mask]

    # PAE is a matrix with size of [L, L], representing global confidence across a sequence
    # Here we use the mean of the matrix as the global confidence score
    return pae.mean()


def from_sequences_to_pdb_files(
    tokenizer,
    model: EsmForProteinFolding,
    sequences: List[str],
    output_dir: str,
) -> Dict[str, str]:
    tokenized_input = tokenizer(
        sequences,
        return_tensors="pt",
        padding=True,
        truncation=True,
        add_special_tokens=False,
        max_length=2048,
    )["input_ids"].to(model.device)
    pae_scores = []
    with torch.no_grad():
        output = model(tokenized_input)
        pae_scores.append(get_pae(output))

    pdbs = model.output_to_pdb(output)

    md5_sequences = [get_md5_sequence(seq) for seq in sequences]
    plddt_scores = []
    for md5_sequence, pdb in zip(md5_sequences, pdbs):
        save_path = f"{output_dir}/{md5_sequence}.pdb"
        with open(save_path, "w") as f:
            f.write(pdb)

        struct = bsio.load_structure(save_path, extra_fields=["b_factor"])
        plddt_scores.append(struct.b_factor.mean())

    ret = []
    for sequence, md5_sequence, plddt, pae in zip(
        sequences, md5_sequences, plddt_scores, pae_scores
    ):
        ret.append(
            {
                "sequence": sequence,
                "filename": f"{md5_sequence}.pdb",
                "plddt": plddt,
                "pae": pae.mean(),
            }
        )
    return ret


def report_foldability_metric(values: List[float], key: str = "plddt"):
    mean_value = np.mean(values)

    if key == "plddt":
        percentile_better_than_threshold = np.mean(np.array(values) > 0.7) * 100
    elif key == "pae":
        percentile_better_than_threshold = (
            np.mean(np.array(values) < 10.0) * 100
        )
    else:
        raise ValueError(f"Invalid key: {key}")

    return {
        "mean": mean_value,
        "percentile_better_than_threshold": percentile_better_than_threshold,
    }


def boxplot(evaluation_dir: str, key: str = "plddt"):
    json_files = []
    for root, _, files in os.walk(evaluation_dir):
        for file in files:
            if file == "foldability.json":
                json_files.append(os.path.join(root, file))

    # sort json_files by category
    json_files = sorted(json_files, key=lambda x: x.split("/")[-2])

    data = []
    labels = []
    for file in json_files:
        category = file.split("/")[-2].split(".")[0]

        with open(file, "r") as f:
            json_data = json.load(f)

        values = [sample[key] for sample in json_data if key in sample]
        data.append(values)
        labels.append(category)

    plt.figure(figsize=(10, 6))
    plt.boxplot(
        data,
        labels=labels,
        patch_artist=True,
        boxprops=dict(facecolor="#73aedf", alpha=0.7),
    )

    plt.title(key.upper(), fontsize=16)
    plt.ylabel(key.upper(), fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(
        os.path.join(evaluation_dir, f"boxplot_{key}.pdf"), format="pdf"
    )


def _main(uid: int, queue: mp.Queue, subset: list, output_pdb_dir: str):
    esmfold_path = "/home/nwliu/data/pretrain/esmfold_v1"
    tokenizer = EsmTokenizer.from_pretrained(esmfold_path)
    model = EsmForProteinFolding.from_pretrained(esmfold_path).to(f"cuda:{uid}")
    model.esm = model.esm.float()
    model.trunk.set_chunk_size(64)

    results = []
    for item in tqdm(
        subset,
        desc=f"Process {uid} - Foldability",
        position=uid + 1,
        ncols=100,
    ):
        sequence = item["response"]
        try:
            res = from_sequences_to_pdb_files(
                tokenizer, model, [sequence], output_pdb_dir
            )
            results.extend(res)
        except Exception as e:
            if str(e).startswith("CUDA out of memory."):
                print(f"CUDA out of memory for {sequence[:20]}")
                continue
            print(e)
            continue

    queue.put(results)


def main(
    num_workers: int,
    sequence_file: str = None,
    evaluation_file: str = None,
    output_pdb_dir: str = None,
    save_plot: bool = False,
    evaluation_dir: str = None,
):
    assert sequence_file and evaluation_file and output_pdb_dir

    if not os.path.exists(evaluation_file):
        mp.set_start_method("spawn", force=True)

        if not os.path.exists(output_pdb_dir):
            os.makedirs(output_pdb_dir)

        with open(sequence_file, "r") as f:
            data = json.load(f)

        queue = mp.Queue()
        processes = []
        results = []
        for i in range(num_workers):
            begin_idx = i * (len(data) // num_workers)
            end_idx = (
                (i + 1) * (len(data) // num_workers)
                if i != num_workers - 1
                else len(data)
            )
            subset = data[begin_idx:end_idx]
            p = mp.Process(
                target=_main, args=(i, queue, subset, output_pdb_dir)
            )
            p.start()
            processes.append(p)

        for _ in processes:
            ret = queue.get()
            results.extend(ret)

        for p in processes:
            p.join()

        with open(evaluation_file, "w") as f:
            json.dump(results, f, indent=4)

    else:
        print("Load processed evaluation file")
        with open(evaluation_file, "r") as f:
            results: list = json.load(f)

    for key in ["plddt", "pae"]:
        print(
            report_foldability_metric([sample[key] for sample in results], key)
        )

    if save_plot:
        for key in ["plddt", "pae"]:
            boxplot(evaluation_dir, key)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
