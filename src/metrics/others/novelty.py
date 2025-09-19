import multiprocessing as mp
import os
import shutil
import subprocess
import tempfile
from datetime import datetime
from typing import Tuple

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from src.configs.others_args import Novelty
from src.utils.folding import seq_to_md5
from src.utils.multiprocess import multiprocess_evaluate

from ..metric import BaseEvaluator, BaseMetric


def compute_structure_novelty(
    sequences: list[str],
    pdb_cache_dir: str,
    target_db: str,
    foldseek_path: str,
    threads: int,
) -> dict[str, Tuple[float, float, list[float]]]:
    with tempfile.TemporaryDirectory() as temp_folder:
        # Prepare Folders
        # query
        query_folder = os.path.join(temp_folder, "query")
        os.mkdir(query_folder)
        query_pdb_folder = os.path.join(query_folder, "pdbs")
        os.mkdir(query_pdb_folder)
        query_db = os.path.join(query_folder, "query")
        # output
        output_folder = os.path.join(temp_folder, "output")
        os.mkdir(output_folder)
        output_db = os.path.join(output_folder, "output")
        results_file = os.path.join(output_folder, "result.tsv")

        seq2strucnov = {}
        # move pdbs to query_folder
        for seq in sequences:
            if os.path.exists(
                os.path.join(pdb_cache_dir, f"{seq_to_md5(seq)}.pdb")
            ):
                shutil.copyfile(
                    os.path.join(pdb_cache_dir, f"{seq_to_md5(seq)}.pdb"),
                    os.path.join(query_pdb_folder, f"{seq_to_md5(seq)}.pdb"),
                )
            else:
                seq2strucnov[seq_to_md5(seq)] = float("nan"), float("nan"), []

        # region foldseek search
        # create query db
        cmd = [
            foldseek_path,
            "createdb",
            query_pdb_folder,
            query_db,
            "--gpu",
            "1",
            "--threads",
            f"{threads}",
            "-v",
            "1",
        ]
        res = subprocess.run(cmd)
        if res.returncode != 0:
            raise RuntimeError("foldseek creadb failed")

        # search
        cmd = [
            foldseek_path,
            "search",
            query_db,
            target_db,
            output_db,
            temp_folder,
            "--gpu",
            "1",
            "--max-seqs",
            "300",
            "-v",
            "1",
            "--threads",
            f"{threads}",
            "-e",
            "100",
            "-a",
        ]
        res = subprocess.run(cmd)
        if res.returncode != 0:
            raise RuntimeError("foldseek search failed")

        # convertalis
        cmd = [
            foldseek_path,
            "convertalis",
            query_db,
            target_db,
            output_db,
            results_file,
            "-v",
            "1",
            "--threads",
            f"{threads}",
            "--format-output",
            "query,alntmscore",
        ]
        res = subprocess.run(cmd)
        if res.returncode != 0:
            raise RuntimeError("foldseek convertalis failed")
        # endregion

        # process result
        matches = pd.read_csv(results_file, sep="\t", header=None)
        matches.columns = ["Query", "TMScore"]
        for seq in sequences:
            query_matches = matches[matches["Query"] == seq_to_md5(seq)]
            if query_matches.empty:
                noveltyH, noveltyE = 1.0, 1.0
            else:
                novelties = query_matches["TMScore"].map(lambda x: 1 - x)
                noveltyH, noveltyE = (
                    novelties.min(),
                    ((300 - len(novelties)) * 1.0 + novelties.sum()) / 300,
                )
            seq2strucnov[seq_to_md5(seq)] = (
                noveltyH,
                noveltyE,
                novelties.to_list(),
            )

        return seq2strucnov


def compute_sequence_novelty(
    sequences: list[str],
    targtedb: str,
    mmseqs_path: str,
    threads: int,
) -> dict[str, Tuple[float, float, list[float]]]:
    """
    compute novelty using mmseq2, modified from [PAAG](https://github.com/chaohaoyuan/PAAG/tree/main/evaluation/unconditional/novelty)

    :param str sequence: protein sequence used to compute novelty
    :param str temp_folder: folder reserved for temporary files
    :param str database_path: path to dataset used by mmseq2
    :param str mmseqs_path: path to mmseq2 executable
    :return float: novelty of the sequence
    """
    with tempfile.TemporaryDirectory() as temp_folder:
        fasta = os.path.join(temp_folder, "temp.fasta")
        querydb = os.path.join(temp_folder, "temp")
        outputdb = os.path.join(temp_folder, "temp.m8")
        result_file = os.path.join(temp_folder, "result.tsv")

        with open(fasta, "w") as f:
            for seq in sequences:
                f.write(f">{seq_to_md5(seq)}\n{seq}\n")

        # fasta to db
        cmd = [
            mmseqs_path,
            "createdb",
            fasta,
            querydb,
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
            querydb,
            targtedb,
            outputdb,
            temp_folder,
            "--gpu",
            "1",
            "--max-seqs",
            "300",
            "-v",
            "1",
            "--threads",
            f"{threads}",
            "-e",
            "100",
        ]
        res = subprocess.run(cmd)
        if res.returncode != 0:
            raise RuntimeError("mmseqs search failed")

        # mmseqs comvertalis
        cmd = [
            mmseqs_path,
            "convertalis",
            querydb,
            targtedb,
            outputdb,
            result_file,
            "-v",
            "1",
            "--threads",
            f"{threads}",
            "--format-output",
            "query,fident",
        ]
        res = subprocess.run(cmd)
        if res.returncode != 0:
            raise RuntimeError("mmseqs convertalis failed")

        seq2seqNov = {}
        matches = pd.read_csv(result_file, sep="\t", header=None)
        matches.columns = ["Query", "Identity"]
        for seq in sequences:
            query_matches = matches[matches["Query"] == seq_to_md5(seq)]
            if query_matches.empty:
                noveltyH, noveltyE = 1.0, 1.0
            else:
                novelties = query_matches["Identity"].map(lambda x: 1 - x)
                noveltyH, noveltyE = (
                    novelties.min(),
                    ((300 - len(novelties)) * 1.0 + novelties.sum()) / 300,
                )
            seq2seqNov[seq_to_md5(seq)] = (
                noveltyH,
                noveltyE,
                [] if novelties.empty else novelties.to_list(),
            )

        return seq2seqNov


def novelty_evaluate_worker(
    queue: mp.Queue,
    pid: int,
    subset: list,
    design_batch_size: int,
    compute_novelties: list[Novelty],
    mmseqs_ex_path: str,
    foldseek_ex_path: str,
    worker_per_mmseqs: int,
    worker_per_foldseek: int,
    mmseqs_targetdb_path: str,
    foldseek_targetdb_path: str,
    pdb_cache_dir: str,
) -> None:
    sequences = list(
        set(
            [
                item[f"response#{b}"]
                for item in subset
                for b in range(1, design_batch_size + 1)
            ]
        )
    )
    if Novelty.Sequence.name in compute_novelties:
        print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        print(f"Compute Seqeuncial Novelty for {len(sequences)} sequences")
        seq2seq_novelty = compute_sequence_novelty(
            sequences=sequences,
            mmseqs_path=mmseqs_ex_path,
            targtedb=mmseqs_targetdb_path,
            threads=worker_per_mmseqs,
        )
    if Novelty.Structure.name in compute_novelties:
        print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        print(f"Compute Structural Novelty for {len(sequences)} sequences")
        seq2struc_novelty = compute_structure_novelty(
            sequences=sequences,
            pdb_cache_dir=pdb_cache_dir,
            target_db=foldseek_targetdb_path,
            foldseek_path=foldseek_ex_path,
            threads=worker_per_foldseek,
        )
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("Finish Novelty Calculation")

    results = []
    for idx, item in enumerate(
        tqdm(
            subset,
            ncols=100,
            disable=True,
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
            md5seq = seq_to_md5(item[f"response#{b}"])
            if Novelty.Sequence.name in compute_novelties:
                if md5seq in seq2seq_novelty:
                    noveltyH, noveltyE, novelties = seq2seq_novelty[md5seq]
                else:
                    noveltyH, noveltyE, novelties = 1.0, 1.0, []
                res[f"Novelty-Hard(Seq)#{b}"] = noveltyH
                res[f"Novelty-Easy(Seq)#{b}"] = noveltyE
                res[f"Novelties(Seq)#{b}"] = novelties

            if Novelty.Structure.name in compute_novelties:
                if md5seq not in seq2struc_novelty:
                    noveltyH, noveltyE, novelties = seq2struc_novelty[md5seq]
                else:
                    noveltyH, noveltyE, novelties = 1.0, 1.0, []
                res[f"Novelty-Hard(Struc)#{b}"] = noveltyH
                res[f"Novelty-Easy(Struc)#{b}"] = noveltyE
                res[f"Novelties(Struc)#{b}"] = novelties

        results.append(res)

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

    def summary(self, results: pd.DataFrame) -> dict:
        _summary = {}
        bs = self.design_batch_size
        if bs == 1:
            if Novelty.Sequence.name in self.novelties:
                _summary["Novelty-Easy(Seq)"] = (
                    results["Novelty-Easy(Seq)#1"].mean() * 100
                )
                _summary["Novelty-Hard(Seq)"] = (
                    results["Novelty-Hard(Seq)#1"].mean() * 100
                )
            if Novelty.Structure.name in self.novelties:
                _summary["Novelty-Easy(Struc)"] = (
                    results["Novelty-Easy(Struc)#1"].mean() * 100
                )
                _summary["Novelty-Hard(Struc)"] = (
                    results["Novelty-Hard(Struc)#1"].mean() * 100
                )
        else:
            if Novelty.Sequence.name in self.novelties:
                easy_novelties = [
                    results[f"Novelty-Easy(Seq)#{b}"].mean() * 100
                    for b in range(1, bs + 1)
                ]
                _summary["Novelty-Easy(Seq)"] = np.nanmean(easy_novelties)
                _summary.update(
                    {
                        f"Novelty-Easy(Seq)#{b}": easy_novelties[b - 1]
                        for b in range(1, bs + 1)
                    }
                )

                hard_novelties = [
                    results[f"Novelty-Hard(Seq)#{b}"].mean() * 100
                    for b in range(1, bs + 1)
                ]
                _summary["Novelty-Hard(Seq)"] = np.nanmean(hard_novelties)
                _summary.update(
                    {
                        f"Novelty-Hard(Seq)#{b}": hard_novelties[b - 1]
                        for b in range(1, bs + 1)
                    }
                )
            if Novelty.Structure.name in self.novelties:
                easy_novelties = [
                    results[f"Novelty-Easy(Struc)#{b}"].mean() * 100
                    for b in range(1, bs + 1)
                ]
                _summary["Novelty-Easy(Seq)"] = np.nanmean(easy_novelties)
                _summary.update(
                    {
                        f"Novelty-Easy(Seq)#{b}": easy_novelties[b - 1]
                        for b in range(1, bs + 1)
                    }
                )

                hard_novelties = [
                    results[f"Novelty-Hard(Struc)#{b}"].mean() * 100
                    for b in range(1, bs + 1)
                ]
                _summary["Novelty-Hard(Seq)"] = np.nanmean(hard_novelties)
                _summary.update(
                    {
                        f"Novelty-Hard(Seq)#{b}": hard_novelties[b - 1]
                        for b in range(1, bs + 1)
                    }
                )
        return _summary


class NoveltyEvaluator(BaseEvaluator):
    def __init__(self, config):
        super().__init__(config)
        self.mmseqs_ex_path = config.novelty.mmseqs_ex_path
        self.foldseek_ex_path = config.novelty.foldseek_ex_path
        self.novelties = config.novelty.novelties
        self.mmseqs_targetdb_path = config.novelty.mmseqs_targetdb_path
        self.foldseek_targetdb_path = config.novelty.foldseek_targetdb_path
        self.workers_per_mmseqs = config.novelty.workers_per_mmseqs
        self.workers_per_foldseek = config.novelty.workers_per_foldseek
        self.pdb_cache_dir = config.novelty.pdb_cache_dir
        self._name = config.novelty.name

    def _excete_manual_multiprocess(self) -> None:
        results = multiprocess_evaluate(
            dataset=self.dataset,  # type: ignore
            eval_worker=novelty_evaluate_worker,
            num_workers=1,
            kwargs={
                "design_batch_size": self.design_batch_size,
                "compute_novelties": self.novelties,
                "mmseqs_ex_path": self.mmseqs_ex_path,
                "foldseek_ex_path": self.foldseek_ex_path,
                "worker_per_mmseqs": self.workers_per_mmseqs,
                "worker_per_foldseek": self.workers_per_foldseek,
                "mmseqs_targetdb_path": self.mmseqs_targetdb_path,
                "foldseek_targetdb_path": self.foldseek_targetdb_path,
                "pdb_cache_dir": self.pdb_cache_dir,
            },
        )
        self.to_json(results)

    def execute(self) -> None:
        self._excete_manual_multiprocess()
