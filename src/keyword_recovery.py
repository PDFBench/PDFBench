import json
import multiprocessing as mp
import os
import shutil
import subprocess
import tempfile
import warnings
from typing import Dict, List, Optional

import numpy as np


def get_keywords(
    sequences: List[str], num_workers: int, interproscan_path: str
) -> Dict[str, List[str]]:
    """
    _summary_

    :param List[str] sequences: _description_
    :param int num_workers: _description_
    :param str interproscan_path: _description_
    :param str temp_folder: _description_
    :return Dict[str, List[str]]: _description_
    """
    with tempfile.TemporaryDirectory() as temp_folder:
        temp_fasta_file = os.path.join(
            temp_folder,
            "temp.fasta",
        )
        temp_output_file = os.path.join(
            temp_folder,
            "temp.json",
        )
        with open(temp_fasta_file, "w") as f:
            for idx, seq in enumerate(sequences):
                seq = "\n".join(
                    [seq[_ : _ + 60] for _ in range(0, len(seq), 60)]
                )
                f.write(f">{idx}\n{seq}\n")

        args = [
            "-i",
            temp_fasta_file,
            "-o",
            temp_output_file,
            "-cpu",
            f"{num_workers}",
            "-f",
            "JSON",
        ]
        print(" ".join([interproscan_path] + args))
        tmp = subprocess.run(
            args=[interproscan_path] + args,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if tmp.returncode != 0:
            warnings.warn(f"InterProScan failed with {sequences[0]}")

        with open(temp_output_file, "r") as f:
            scan_results = json.load(f)["results"]

    # process the result
    results = {}
    for scan_result in scan_results:
        sequence = scan_result["sequence"]
        results[sequence] = []
        seen = set()
        for match in scan_result["matches"]:
            try:
                interpro_id = match["signature"]["entry"]["accession"]
                interpro_name = match["signature"]["entry"]["name"]
            except (TypeError, KeyError):
                continue
            for location in match["locations"]:
                beg = location["start"]
                end = location["end"]
                pattern = (interpro_id, beg, end)
                if pattern not in seen:
                    seen.add(pattern)
                    results[sequence].append(
                        {
                            "InterPro-ID": interpro_id,
                            "InterPro-Name": interpro_name,
                            "Beg": location["start"],
                            "End": location["end"],
                        }
                    )
    return results


def compute_keyword_recovery(
    references: List[str],
    responses: List[str],
    num_workers: int,
    interpro_scan_path: str,
) -> list[dict]:
    """
    implementation of ESM3 keyword recovery, We report function keyword recovery
    at the protein level, computing the proportion of all function keywords in
    the prompt which appear anywhere in the function keywords from the
    InterProScan annotations of the generation.

    :param str reference: _description_
    :param str response: _description_
    :param str interpro_scan_path: _description_
    :param str temp_folder: _description_
    :return float: _description_
    """
    assert len(references) == len(responses)
    print(
        f"{len(references)} sequences, nearly {len(references) / 450:.3f} hours"
    )
    keywords = get_keywords(
        references + responses, num_workers, interpro_scan_path
    )

    try:
        ref_seq_ids_lst = [
            {
                "seqeunce": seq,
                "keywords": list(
                    set([item["InterPro-ID"] for item in keywords[seq]])  # type: ignore
                ),
            }
            if seq != ""
            else []
            for seq in references
        ]
        res_seq_ids_lst = [
            {
                "seqeunce": seq,
                "keywords": list(
                    set([item["InterPro-ID"] for item in keywords[seq]])  # type: ignore
                ),
            }
            if seq != ""
            else []
            for seq in responses
        ]
    except KeyError as e:
        for idx, seq in enumerate(responses):
            print(idx, seq[:10])
        warnings.warn(f"Error in computing keyword recovery: \n{e}")
        raise RuntimeError

    res = []
    for ref_pair, res_pair in zip(ref_seq_ids_lst, res_seq_ids_lst):
        ref_ids: dict = ref_pair["keywords"]  # type: ignore
        res_ids: dict = res_pair["keywords"]  # type: ignore
        ref_seq = ref_pair["seqeunce"]  # type: ignore
        res_seq = res_pair["seqeunce"]  # type: ignore
        if len(ref_ids) == 0:
            rec = 0.0
        else:
            rec = sum([id in set(ref_ids) for id in res_ids]) / len(ref_ids)

        res.append(
            {
                "keyword_recovery": rec,
                "reference": ref_seq,
                "response": res_seq,
            }
        )

    return res


def _main(
    uid: int,
    queue: mp.Queue,
    subset: list,
    workers_per_scan: int,
    interpro_scan_path: str,
):
    responses = [item["response"] for item in subset]
    references = [item["reference"] for item in subset]

    results = compute_keyword_recovery(
        references=references,
        responses=responses,
        num_workers=workers_per_scan,
        interpro_scan_path=interpro_scan_path,
    )

    queue.put(results)


def main(
    num_workers: int = 4,
    sequence_file: Optional[str] = None,
    evaluation_file: Optional[str] = None,
    workers_per_scan: int = 4,
    interpro_scan_path: Optional[str] = None,
):
    assert sequence_file and evaluation_file

    try:
        assert shutil.which("java") is not None
    except AssertionError:
        print(
            "Java not found. Please install Java 11 and place it on your path, and run the script again."
        )
        return

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
                    subset,
                    workers_per_scan,
                    interpro_scan_path,
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
        "keyword_recovery",
    ]
    for metric in support_metrics:
        mean = np.mean([sample[metric] for sample in results])
        print(f"mean {metric}: {mean}")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
