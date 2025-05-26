import json
import multiprocessing as mp
import os
import subprocess
import tempfile
import warnings

import numpy as np
from tqdm.auto import tqdm


def compute_novelty(
    sequence: str,
    # temp_folder: str,
    database_path: str,
    mmseqs_path: str,
) -> float:
    """
    compute novelty using mmseq2, copied from [PAAG](https://github.com/chaohaoyuan/PAAG/tree/main/evaluation/unconditional/novelty)

    :param str sequence: protein sequence used to compute novelty
    :param str temp_folder: folder reserved for temporary files
    :param str database_path: path to dataset used by mmseq2
    :param str mmseqs_path: path to mmseq2 executable
    :return float: novelty of the sequence
    """

    def process_m8_file(file_path, n_prot=300):
        max_similarity = {}
        with open(file_path, "r") as file:
            for line in file:
                parts = line.strip().split("\t")
                if len(parts) < 3:
                    continue
                query_id = parts[0]
                similarity = float(parts[2])

                if query_id not in max_similarity:
                    max_similarity[query_id] = similarity
                else:
                    max_similarity[query_id] = max(
                        max_similarity[query_id], similarity
                    )

        # hit
        hits = 0
        for similarity in max_similarity.values():
            hits += 1 - similarity
        # dismiss
        dismisses = (n_prot - len(max_similarity)) * 1.0
        novelty = (hits + dismisses) / n_prot

        return novelty

    def mmseqs_search():
        # if not os.path.exists(temp_folder):
        #     os.makedirs(temp_folder)

        # sequence to fasta
        with open(temp_fasta_file, "w") as f:
            fasta_sequence = "\n".join(
                [sequence[_ : _ + 60] for _ in range(0, len(sequence), 60)]
            )
            f.write(f">temp\n{fasta_sequence}\n")

        # fasta to db
        cmd = [
            mmseqs_path,
            "createdb",
            temp_fasta_file,
            temp_db_file,
            "-v",
            "1",
        ]
        res = subprocess.run(cmd)
        if res.returncode != 0:
            raise RuntimeError("mmseqs creadb failed")

        # mmseqs search
        cmd = [
            mmseqs_path,
            "search",
            temp_db_file,
            database_path,
            temp_output_file,
            temp_folder,
            "--gpu",
            "1",
            "--max-seqs",
            "300",
            "-v",
            "1",
            "--remove-tmp-files",
            "1",
            "--threads",
            "6",
            "-e",
            "100",
        ]
        res = subprocess.run(cmd)

        return res

    # Create a unique temporary folder for this run
    # current = time.strftime("%Y%m%d%H%M%S", time.localtime())
    # temp_folder = os.path.join(
    #     temp_folder, os.uname().nodename, f"{current}_{randint(1, 100)}"
    # )  # temp_folder/node_name/timestamp == temp_node_folder/timestamp
    with tempfile.TemporaryDirectory() as temp_folder:
        temp_fasta_file = os.path.join(temp_folder, "temp.fasta")
        temp_db_file = os.path.join(temp_folder, "temp")
        temp_output_file = os.path.join(temp_folder, "temp.m8")

        error_times = 0
        error_message = ""
        while error_times < 3:
            try:
                mmseqs_search()
                novelty = process_m8_file(temp_output_file, 300)
                break
            except FileNotFoundError:
                warnings.warn("Sequence is too strage to search *_*")
                novelty = 1.0
                break
            except Exception as e:
                error_message = str(e)
            finally:
                error_times += 1
        else:
            novelty = 1.0
            warnings.warn(error_message)

    return novelty


def _main(
    uid: int,
    queue: mp.Queue,
    subset: list,
    mmseqs_path: str,
    database_path: str,
    # temp_folder: str,
    devices: list,
) -> None:
    """
    _summary_:TODO: Add summary

    :param int uid: _description_
    :param mp.Queue queue: _description_
    :param list subset: _description_
    """
    # set the cuda device, enable multi-gpu for mmseqs2
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{devices[uid]}"

    results: list = [dict() for _ in range(len(subset))]
    idx = 0
    for item in tqdm(
        subset,
        desc=f"Process {uid + 1} - Novelty",
        position=uid + 1,
        ncols=100,
    ):
        mutant = item["response"]

        res: dict = {
            "sequence": mutant,
            "novelty": compute_novelty(
                sequence=mutant,
                mmseqs_path=mmseqs_path,
                database_path=database_path,
                # temp_folder=temp_folder,
            ),
        }

        results[idx].update(res)
        idx += 1

    # reset the cuda device
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(devices)

    queue.put(results)


def main(
    sequence_file: str,
    evaluation_file: str,
    mmseqs_path: str,
    database_path: str,
    num_workers: int = 4,
) -> None:
    assert sequence_file and evaluation_file

    if not os.path.exists(evaluation_file):
        mp.set_start_method("spawn", force=True)

        with open(sequence_file, "r") as f:
            data: list = json.load(f)

        # # create temp foler for node, namely all the subprocesses
        # temp_node_folder = os.path.join(temp_folder, os.uname().nodename)
        # if not os.path.exists(temp_node_folder):
        #     os.makedirs(temp_node_folder)

        devices: list = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")

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
                    mmseqs_path,
                    database_path,
                    # temp_folder,
                    devices,
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

        # # remove temp folder for node
        # if os.path.exists(temp_node_folder):
        #     shutil.rmtree(temp_node_folder)
    else:
        print("Load processed evaluation file")
        with open(evaluation_file, "r") as f:
            results: list = json.load(f)

    support_metrics = [
        "novelty",
    ]
    for metric in support_metrics:
        mean = np.mean([sample[metric] for sample in results])
        print(f"mean {metric}: {mean:.2f}")


def test():
    sequence = "MSSSSSGGPPGTVTGTGSGGDGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTETETGTGTGTGTGTGTETETGEGTETEEEEEEEEEEEEEEEEEEEEEEEEEEE"
    compute_novelty(
        sequence=sequence,
        database_path="/home/jhkuang/data/datasets/MMseqs/db/uniprotkb/uniprot_gpu",
        mmseqs_path="/home/jhkuang/app/mmseqs/bin/mmseqs",
    )


if __name__ == "__main__":
    DEBUG = False
    if DEBUG:
        test()
    else:
        import fire

        fire.Fire(main)
