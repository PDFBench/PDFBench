import hashlib
import multiprocessing as mp
import os
import subprocess
import tempfile
from typing import List, Tuple

import numpy as np
from tqdm.auto import tqdm
from transformers import EsmForProteinFolding, EsmTokenizer, logging

from src.configs.others_args import Diversity
from src.utils.folding import seq_to_md5, seq_to_struc
from src.utils.multiprocess import multiprocess_evaluate

from ..metric import BaseEvaluator, BaseMetric

logging.set_verbosity_error()


def get_md5_sequence(sequence: str) -> str:
    return hashlib.md5(sequence.encode()).hexdigest()


def compute_tm_score(
    ref: str,
    res: str,
    tm_score_path: str,
    pdb_cache_dir: str,
    model: EsmForProteinFolding,
    tokenizer: EsmTokenizer,
):
    pdb_ref = os.path.join(pdb_cache_dir, f"{seq_to_md5(ref)}.pdb")
    if not os.path.exists(pdb_ref):
        seq_to_struc(
            tokenizer=tokenizer,
            model=model,
            sequences=[ref],
            pdb_cache_dir=pdb_cache_dir,
            return_foldability=False,
        )

    pdb_res = os.path.join(pdb_cache_dir, f"{seq_to_md5(res)}.pdb")
    if not os.path.exists(pdb_ref):
        seq_to_struc(
            tokenizer=tokenizer,
            model=model,
            sequences=[res],
            pdb_cache_dir=pdb_cache_dir,
            return_foldability=False,
        )

    try:
        result = subprocess.run(
            args=[
                tm_score_path,
                pdb_ref,
                pdb_res,
                "-outfmt",
                "2",  # omit the duplicated output
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=True,
        )
        gt_tmscore_output = result.stdout
        tmscore = gt_tmscore_output.split("\n")[1].split("\t")[3]
        return float(tmscore)
    except Exception:
        return float("nan")


def compute_structure_diversity(
    sequences: List[str],
    pdb_cache_dir: str,
    tm_score_path: str,
    model: EsmForProteinFolding,
    tokenizer: EsmTokenizer,
) -> Tuple[float, list]:
    assert len(sequences) >= 2, (
        "Structural diversity requires at least two sequences."
    )

    diversities = []
    for idx in range(len(sequences)):
        for idy in range(idx + 1, len(sequences)):
            sim_struct = compute_tm_score(
                ref=sequences[idx],
                res=sequences[idy],
                tm_score_path=tm_score_path,
                pdb_cache_dir=pdb_cache_dir,
                model=model,
                tokenizer=tokenizer,
            )
            diversities.append([1.0 - sim_struct, idx + 1, idy + 1])

    if len(diversities) == 0:
        return float("nan"), []
    else:
        diversity = np.array(diversities)[:, 0].mean()
        return diversity, diversities


def process_m8_file(file_path, n_prot=3) -> Tuple[float, list[float]]:
    diversities = []
    with open(file_path, "r") as file:
        for line in file:
            parts = line.strip().split("\t")
            assert len(parts) > 3, "MMSeqs2 M8 should have at least 3 columns"
            query_id, match_id = parts[0], parts[1]
            if query_id == match_id:
                continue

            similarity = float(parts[2])
            diversities.append(1.0 - similarity)

    total = n_prot * (n_prot - 1)
    diversities.extend([1.0] * (total - len(diversities)))

    diversity = sum(diversities) / len(diversities)

    return diversity, diversities


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


def compute_sequence_diversity(
    sequences: List[str],
    mmseqs_path: str,
) -> Tuple[float, list]:
    """
    Computes diversity within a list of sequences using MMseqs2.

    Diversity is defined as the mean dissimilarity (1 - similarity)
    between all unique pairs of sequences in the input list.

    :param List[str] sequences: A list of protein sequences.
    :param str temp_folder_base: Base directory for temporary files.
                                 A unique subfolder will be created inside this.
    :param str mmseqs_path: Path to the MMseqs2 executable.
    :return float: The computed diversity value (average dissimilarity).
                   Returns 0.0 if less than 2 sequences are provided.
    """
    assert len(sequences) >= 2, "Diversity requires at least two sequences."

    with tempfile.TemporaryDirectory() as temp_folder:
        fasta_file = os.path.join(temp_folder, "sequences.fasta")
        result_m8_file = os.path.join(temp_folder, "result.m8")

        res = mmseqs_easy_search(
            mmseqs_path,
            sequences,
            fasta_file,
            result_m8_file,
            temp_folder,
        )
        if res.returncode != 0:
            return float("nan"), []

        diversity, diversities = process_m8_file(
            result_m8_file, n_prot=len(sequences)
        )

    return diversity, diversities


def diversity_evaluate_worker(
    queue: mp.Queue,
    pid: int,
    subset: list,
    design_batch_size: int,
    diversities: list[Diversity],
    mmseqs_ex_path: str,
    tm_score_ex_path: str,
    esm_fold_name_or_path: str,
    pdb_cache_dir: str,
) -> None:
    if (
        design_batch_size is None
        or diversities is None
        or mmseqs_ex_path is None
        or mmseqs_ex_path is None
        or tm_score_ex_path is None
        or esm_fold_name_or_path is None
    ):
        raise ValueError(
            "Invalid kwargs: \n"
            f"design_batch_size: {design_batch_size}\n"
            f" diveristies: {diversities}\n"
            f"mmseqs_ex_path: {mmseqs_ex_path}\n"
            f"tm_score_ex_path: {tm_score_ex_path}\n"
            f"esm_fold_name_or_path: {esm_fold_name_or_path}"
        )

    if Diversity.Structure.name in diversities:
        tokenizer = EsmTokenizer.from_pretrained(esm_fold_name_or_path)
        model: EsmForProteinFolding = EsmForProteinFolding.from_pretrained(
            esm_fold_name_or_path
        ).to(f"cuda:{pid}")  # type: ignore
        model.esm = model.esm.float()
        model.trunk.set_chunk_size(64)  # type: ignore

    results = []
    for idx, item in enumerate(
        tqdm(
            subset,
            desc="Diversity",
            ncols=100,
            disable=pid != 0,
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

        responses = [
            item[f"response#{b}"] for b in range(1, design_batch_size + 1)
        ]

        if Diversity.Sequence.name in diversities:
            try:
                seq_div, seq_divs = compute_sequence_diversity(
                    sequences=responses,
                    mmseqs_path=mmseqs_ex_path,
                )
            except Exception:
                # warnings.warn(f"Diversity Error with {e}")    # TODO: Error Logging
                seq_div = float("nan")
                seq_divs = []
            res.update(
                {
                    "sequence_diversity": seq_div,
                    "sequence_diversities": seq_divs,
                }
            )

        if Diversity.Structure.name in diversities:
            try:
                struct_div, struct_divs = compute_structure_diversity(
                    sequences=responses,
                    pdb_cache_dir=pdb_cache_dir,
                    tm_score_path=tm_score_ex_path,
                    model=model,
                    tokenizer=tokenizer,
                )
            except Exception:
                # warnings.warn(f"Diversity Error with {e}")    # TODO: Error Logging
                struct_div = float("nan")
                struct_divs = []
            res.update(
                {
                    "structure_diversity": struct_div,
                    "structure_diversities": struct_divs,
                }
            )

        results.append(res)

    queue.put((pid, results))


class DiversityMetric(BaseMetric):
    def __init__(self, config):
        super().__init__(config)
        self._name = config.diversity.name
        self.mmseqs_ex_path = config.diversity.mmseqs_ex_path
        self.tm_score_ex_path = config.diversity.tm_score_ex_path
        self.pdb_cache_dir = config.diversity.pdb_cache_dir
        self.esm_fold_name_or_path = config.diversity.esm_fold_name_or_path
        self.diversities = config.diversity.diversities

    @property
    def metrics(self) -> list[str]:
        _metrics = []
        if Diversity.Sequence.name in self.diversities:
            _metrics.extend("Diversity(Seq)")
        if Diversity.Structure.name in self.diversities:
            _metrics.append("Diversity(Struc)")
        return _metrics

    def summary(self, results) -> dict[str, float]:
        _summary = {}
        if Diversity.Sequence.name in self.diversities:
            _summary["Diversity(Seq)"] = (
                results["sequence_diversity"].mean() * 100
            )
        if Diversity.Structure.name in self.diversities:
            _summary["Diversity(Struc)"] = (
                results["structure_diversity"].mean() * 100
            )
        return _summary


class DiversityEvaluator(BaseEvaluator):
    def __init__(self, config):
        super().__init__(config)
        self._name = config.diversity.name
        self.mmseqs_ex_path = config.diversity.mmseqs_ex_path
        self.tm_score_ex_path = config.diversity.tm_score_ex_path
        self.pdb_cache_dir = config.diversity.pdb_cache_dir
        self.esm_fold_name_or_path = config.diversity.esm_fold_name_or_path
        self.diversities = config.diversity.diversities

    def _execute_manual_multiprocess(self) -> None:
        results = multiprocess_evaluate(
            dataset=self.dataset,
            eval_worker=diversity_evaluate_worker,
            num_workers=self.num_gpu,
            kwargs={
                "design_batch_size": self.design_batch_size,
                "diversities": self.diversities,
                "mmseqs_ex_path": self.mmseqs_ex_path,
                "tm_score_ex_path": self.tm_score_ex_path,
                "esm_fold_name_or_path": self.esm_fold_name_or_path,
                "pdb_cache_dir": self.pdb_cache_dir,
            },
        )
        self.to_json(results)

    def execute(self) -> None:
        self._execute_manual_multiprocess()
