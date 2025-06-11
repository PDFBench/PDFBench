import hashlib
import json
import multiprocessing as mp
import os
import subprocess
import warnings

import numpy as np
from tqdm.auto import tqdm


def get_md5_sequence(sequence: str) -> str:
    return hashlib.md5(sequence.encode()).hexdigest()


def compute_tm_score(
    ref: str, res: str, tm_score_path: str, output_pdb_dir: str
):
    md5_ref = get_md5_sequence(ref)
    md5_res = get_md5_sequence(res)
    pdb_ref = os.path.join(output_pdb_dir, f"{md5_ref}.pdb")
    pdb_res = os.path.join(output_pdb_dir, f"{md5_res}.pdb")
    assert os.path.exists(pdb_ref), f"PDB ref: {pdb_ref}"
    assert os.path.exists(pdb_res), f"PDB res: {pdb_res}"
    args = [
        tm_score_path,
        pdb_ref,
        pdb_res,
        "-outfmt",
        "2",  # omit the duplicated output
    ]

    try:
        result = subprocess.run(
            args=args,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=True,
        )
        gt_tmscore_output = result.stdout
        tmscore = gt_tmscore_output.split("\n")[1].split("\t")[3]
        return float(tmscore)
    except Exception as e:
        warnings.warn(f"TmScore Error with {e}")
        return 0.0


def _main(
    uid: int,
    queue: mp.Queue,
    subset: list,
    tm_score_path: str,
    output_pdb_dir: str,
):
    assert os.listdir(output_pdb_dir), (
        "PDB fies do not exists, please run the foldability first."
    )
    assert os.path.exists(tm_score_path), (
        "TMScore does not exist, please check."
    )

    results: list = [dict() for _ in range(len(subset))]
    num_error = 0
    par = tqdm(
        subset,
        desc=f"Process {uid} - TMScore",
        position=uid + 1,
        ncols=100,
    )
    par.set_postfix(erro_ratio=0.0)
    for idx, item in enumerate(par):
        reference = item["reference"]
        response = item["response"]
        try:
            res = {
                "reference": reference,
                "response": response,
                "tm_score": compute_tm_score(
                    reference, response, tm_score_path, output_pdb_dir
                ),
            }
            results[idx].update(res)
        except AssertionError:
            # print(f"Ref[{reference[:20]}], Res[{response[:20]}] Missed")
            num_error += 1
            par.set_postfix(erro_ratio=num_error / len(subset))
            continue
    queue.put(results)


def main(
    num_workers: int,
    sequence_file: str,
    evaluation_file: str,
    output_pdb_dir: str,
    tm_score_path: str,
):
    assert sequence_file and evaluation_file and output_pdb_dir

    if not os.path.exists(evaluation_file):
        mp.set_start_method("spawn", force=True)

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
                target=_main,
                args=(i, queue, subset, tm_score_path, output_pdb_dir),
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
        "tm_score",
    ]
    for metric in support_metrics:
        mean = np.mean(
            [sample[metric] for sample in results if metric in sample]
        )
        print(f"mean {metric}: {mean:.3f}")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
