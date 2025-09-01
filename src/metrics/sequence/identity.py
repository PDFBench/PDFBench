import multiprocessing as mp
import os
import subprocess
import tempfile

from tqdm.auto import tqdm

from src.datasets import BaseDataset
from src.metrics import BaseMetric
from src.utils.multiprocess import multiprocess_evaluate


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
    thread: int,
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
        f"{thread}",
        "-e",
        "1000000",
    ]
    return subprocess.run(args)


def compute_identity(
    wild: str, mutant: str, mmseqs_ex_path: str, thread_per_mmseqs: int
) -> float:
    with tempfile.TemporaryDirectory() as temp_folder:
        fasta_file = os.path.join(temp_folder, "sequences.fasta")
        result_m8_file = os.path.join(temp_folder, "result.m8")

        res = mmseqs_easy_search(
            mmseqs_ex_path,
            [mutant, wild],
            fasta_file,
            result_m8_file,
            temp_folder,
            thread_per_mmseqs,
        )
        if res.returncode != 0:
            raise RuntimeError("mmseqs easy-search failed")
        identity = process_m8_file(result_m8_file, n_prot=2)

    return identity


def identity_evaluate_worker(
    queue: mp.Queue, pid: int, subset: list, **kwargs
) -> None:
    design_batch_size = kwargs.get("design_batch_size")
    verbose = kwargs.get("verbose")
    mmseqs_ex_path = kwargs.get("mmseqs_ex_path")
    thread_per_mmseqs = kwargs.get("thread_per_mmseqs")
    if not (
        design_batch_size is not None
        and verbose is not None
        and mmseqs_ex_path is not None
        and thread_per_mmseqs is not None
    ):
        raise ValueError(
            "Invalid kwargs: "
            f"{design_batch_size}, {verbose}, {mmseqs_ex_path}, {thread_per_mmseqs}"
        )

    items = tqdm(
        subset,
        desc="Identity",
        position=pid + 1,
        ncols=100,
        disable=not verbose or pid != 0,
    )
    results: list = [dict() for _ in range(len(subset))]
    for idx, item in enumerate(items):
        res = {
            "instruction": item["instruction"],
            "reference": item["reference"],
        }
        for b in range(1, design_batch_size + 1):
            reference = item["reference"]
            response = item[f"response#{b}"]

            res.update(
                {
                    f"response#{b}": response,
                    f"identity#{b}": compute_identity(
                        reference, response, mmseqs_ex_path, thread_per_mmseqs
                    ),
                }
            )
        results[idx].update(res)

    queue.put(results)


class IdentityMetric(BaseMetric):
    def __init__(self, config):
        super().__init__(config)
        self.mmseqs_ex_path = config.identity.mmseqs_ex_path
        self.thread_per_mmseqs = config.identity.thread_per_mmseqs
        self._name = config.identity.name

    @property
    def metrics(self) -> list[str]:
        return ["identity"]

    def _evaluate(
        self,
        dataset: BaseDataset,
    ) -> list[dict]:
        return multiprocess_evaluate(
            dataset=dataset,
            eval_worker=identity_evaluate_worker,
            num_workers=self.num_cpu // self.thread_per_mmseqs,
            kwargs={
                "design_batch_size": self.design_batch_size,
                "verbose": self.verbose,
                "thread_per_mmseqs": self.thread_per_mmseqs,
                "mmseqs_ex_path": self.mmseqs_ex_path,
            },
        )
