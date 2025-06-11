import json
import multiprocessing as mp
import os
import subprocess
import tempfile

import numpy as np
from tqdm.auto import tqdm


def process_m8_file(file_path, n_prot=2):
    similaritys = []
    with open(file_path, "r") as file:
        for line in file:
            parts = line.strip().split("\t")
            assert len(parts) > 3, "MMSeqs2 M8 should have at least 3 columns"
            query_id, match_id = parts[0], parts[1]
            if query_id == match_id:
                continue

            similarity = float(parts[2])
            similaritys.append(similarity)

    total = n_prot * (n_prot - 1)
    hits = sum(similaritys)
    dismiss = (total - len(similaritys)) * 0.0
    similarity = (hits + dismiss) / total

    return similarity


def mmseqs_easy_search(
    mmseqs_path: str,
    sequences: list[str],
    fasta_file: str,
    result_m8_file: str,
    temp_folder: str,
):
    # --- 1. Prepare FASTA file ---
    with open(fasta_file, "w") as f:
        for i, seq in enumerate(sequences):
            f.write(f">seq_{i}\n")
            fasta_sequence = "\n".join(
                [seq[j : j + 60] for j in range(0, len(seq), 60)]
            )
            f.write(f"{fasta_sequence}\n")

    # --- 2. Run MMseqs2 ---
    args = [
        mmseqs_path,
        "easy-search",
        fasta_file,
        fasta_file,
        result_m8_file,
        temp_folder,
        "-v",
        "1",
        "--remove-tmp-files",
        "1",
        "--threads",
        "6",
        "-e",
        "1000000",
    ]
    return subprocess.run(args)


def compute_identity(wild: str, mutant: str, mmseqs_path: str) -> float:
    with tempfile.TemporaryDirectory() as temp_folder:
        fasta_file = os.path.join(temp_folder, "sequences.fasta")
        result_m8_file = os.path.join(temp_folder, "result.m8")

        res = mmseqs_easy_search(
            mmseqs_path,
            [mutant, wild],
            fasta_file,
            result_m8_file,
            temp_folder,
        )
        if res.returncode != 0:
            raise RuntimeError("mmseqs easy-search failed")
        identity = process_m8_file(result_m8_file, n_prot=2)

    return identity


def _main(uid: int, queue: mp.Queue, subset: list, mmseqs_path: str):
    results: list = [dict() for _ in range(len(subset))]

    idx = 0
    for item in tqdm(
        subset,
        desc=f"Process {uid} - Identity",
        position=uid + 1,
        ncols=100,
    ):
        mutant = item["response"]
        wild_type = item["reference"]

        res: dict = {
            "reference": wild_type,
            "response": mutant,
            "identity": compute_identity(wild_type, mutant, mmseqs_path),
        }
        results[idx].update(res)
        idx += 1

    queue.put(results)


def main(
    num_workers: int, sequence_file: str, evaluation_file: str, mmseqs_path: str
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

            p = mp.Process(target=_main, args=(i, queue, subset, mmseqs_path))
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
        "identity",
    ]
    for metric in support_metrics:
        mean = np.mean([sample[metric] for sample in results])
        print(f"mean {metric}: {mean:.2f}")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
