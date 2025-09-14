import multiprocessing as mp
import os
import subprocess
import tempfile
import warnings

import numpy as np
from pandas import DataFrame
from tqdm.auto import tqdm

from src.configs.others_args import Novelty
from src.utils.multiprocess import multiprocess_evaluate

from ..metric import BaseEvaluator, BaseMetric


def compute_novelty(
    sequence: str,
    database_path: str,
    mmseqs_path: str,
    workers_per_mmseqs: int,
) -> float:
    """
    compute novelty using mmseq2, modified from [PAAG](https://github.com/chaohaoyuan/PAAG/tree/main/evaluation/unconditional/novelty)

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
            "--dbtype",
            "1",
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
            "--threads",
            f"{workers_per_mmseqs}",
            "-e",
            "100",
        ]
        res = subprocess.run(cmd)

        return res

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


def novelty_evaluate_worker(
    queue: mp.Queue,
    pid: int,
    subset: list,
    design_batch_size: int,
    novelties: list[Novelty],
    mmseqs_ex_path: str,
    worker_per_mmseqs: int,
    database_path: str,
    devices: list,
) -> None:
    if (
        design_batch_size is None
        or novelties is None
        or mmseqs_ex_path is None
        or worker_per_mmseqs is None
        or database_path is None
        or devices is None
    ):
        raise ValueError(
            "Invalid kwargs: \n"
            f"design_batch_size: {design_batch_size}\n"
            f"novelties: {novelties}\n"
            f"mmseqs_ex_path: {mmseqs_ex_path}\n"
            f"worker_per_mmseqs: {worker_per_mmseqs}\n"
            f"database_path: {database_path}\n"
            f"devices: {devices}"
        )

    # set the cuda device, enable multi-gpu for mmseqs2
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{devices[pid]}"

    results = []
    for idx, item in enumerate(
        tqdm(
            subset,
            # desc="Novelty",
            desc=f"Process{pid + 1}: Novelty",
            ncols=100,
            position=pid,
            # disable=pid != 0,
        )
    ):
        res = {
            "instruction": item["instruction"],
            "reference": item["reference"],
            **{
                f"response#{b}": item[f"response#{b}"]
                for b in range(1, design_batch_size + 1)
            },
        }
        for b in range(1, design_batch_size + 1):
            res[f"novelty#{b}"] = compute_novelty(
                sequence=item[f"response#{b}"],
                mmseqs_path=mmseqs_ex_path,
                workers_per_mmseqs=worker_per_mmseqs,
                database_path=database_path,
            )
        results.append(res)

    # reset the cuda device
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(devices)

    queue.put((pid, results))


class NoveltyMetric(BaseMetric):
    def __init__(self, config):
        super().__init__(config)
        self.mmseqs_ex_path = config.novelty.mmseqs_ex_path
        self.novelties = config.novelty.novelties
        self._name = config.novelty.name

    @property
    def metrics(self) -> list[str]:
        _metrics = []
        if Novelty.Sequence.name in self.novelties:
            _metrics.extend("Novelty(Seq)")
        if Novelty.Structure.name in self.novelties:
            _metrics.append("Novelty(Struc)")
        return _metrics

    def summary(self, results: DataFrame) -> dict:
        _summary = {}
        bs = self.design_batch_size
        if bs == 1:
            if Novelty.Sequence.name in self.novelties:
                _summary["Novelty(Seq)"] = results["novelty#1"].mean() * 100
            # if Novelty.Structure.name in self.novelties:
            #     _summary["Novelty(Struc)"] = results["novelty#1"].mean()
        else:
            if Novelty.Sequence.name in self.novelties:
                seq_novelties = [
                    results[f"novelty#{b}"].mean() * 100
                    for b in range(1, bs + 1)
                ]
                _summary["Novelty(Seq)"] = np.nanmean(seq_novelties)
                _summary.update(
                    {
                        f"Novelty(Seq)#{b}": seq_novelties[b - 1]
                        for b in range(1, bs + 1)
                    }
                )
            # if Novelty.Structure.name in self.novelties:
            #     seq_novelties = [
            #         results[f"novelty#{b}"].mean() for b in range(1, bs + 1)
            #     ]
            #     _summary["Novelty(Struc)"] = np.nanmean(seq_novelties)
            #     _summary.update(
            #         {
            #             "Novelty(Struc)#{b}": seq_novelties[b - 1]
            #             for b in range(1, bs + 1)
            #         }
            #     )
        return _summary


class NoveltyEvaluator(BaseEvaluator):
    def __init__(self, config):
        super().__init__(config)
        self.mmseqs_ex_path = config.novelty.mmseqs_ex_path
        self.novelties = config.novelty.novelties
        self.database_path = config.novelty.database_path
        self._name = config.novelty.name

    def _excete_manual_multiprocess(self) -> None:
        devices: list = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        results = multiprocess_evaluate(
            dataset=self.dataset,  # type: ignore
            eval_worker=novelty_evaluate_worker,
            num_workers=self.num_gpu,
            kwargs={
                "design_batch_size": self.design_batch_size,
                "novelties": self.novelties,
                "mmseqs_ex_path": self.mmseqs_ex_path,
                "worker_per_mmseqs": self.num_cpu // self.num_gpu - 1,
                "database_path": self.database_path,
                "devices": devices,
            },
        )
        self.to_json(results)

    def execute(self) -> None:
        self._excete_manual_multiprocess()
