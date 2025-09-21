import json
import multiprocessing as mp
import os
import subprocess
import tempfile
from datetime import datetime

import numpy as np
from pandas import DataFrame
from tqdm.auto import tqdm

from src.utils import logging
from src.utils.multiprocess import multiprocess_evaluate

from ..metric import BaseEvaluator, BaseMetric

logger = logging.get_logger(__name__)


def get_keywords(
    sequences: list[str],
    workers_per_scan: int,
    ipr_scan_ex: str,
) -> dict[str, list[str]]:
    with tempfile.TemporaryDirectory() as temp_folder:
        temp_fasta_file = os.path.join(temp_folder, "temp.fasta")
        temp_output_file = os.path.join(temp_folder, "temp.json")
        temp_temp_folder = os.path.join(temp_folder, "temp")
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
            f"{workers_per_scan}",
            "-T",
            temp_temp_folder,
            # "-vl",
            # "0",
            "-f",
            "JSON",
        ]
        out = subprocess.run(
            [ipr_scan_ex] + args,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=None,
        )
        if out.returncode != 0:
            raise Exception("IPR Scan failed")
        with open(temp_output_file, "r") as f:
            scan_results = json.load(f)["results"]

    results = {}
    for scan_result in scan_results:
        sequence = scan_result["sequence"]
        results[sequence] = []
        seen = set()
        for match in scan_result["matches"]:
            try:
                interpro_id = match["signature"]["entry"]["accession"]
                interpro_name = match["signature"]["entry"]["name"]
                interpro_type = match["signature"]["entry"]["type"]
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
                            "InterPro-Type": interpro_type,
                            "Beg": location["start"],
                            "End": location["end"],
                        }
                    )

    return results


def load_dict(
    ipr_scan_ex: str,
    workers_per_scan: int,
    ipr_cache_path: str,
    update_seqs: list[str] | None = None,
):
    if not os.path.exists(ipr_cache_path):
        cache = {}
    else:
        with open(ipr_cache_path, "r") as f:
            cache = json.load(f)
            last_update_date = datetime.fromtimestamp(
                os.path.getmtime(ipr_cache_path)
            ).strftime("%Y-%m-%d %H:%M:%S")

    if cache != {}:
        print(f">>> Load {len(cache)} sequences from cache>>>")
        print(f"Cache last updated at {last_update_date}")

    if update_seqs is not None:
        cache_seqs = set(list(cache.keys()))
        update_seqs = [seq for seq in update_seqs if seq not in cache_seqs]

        if update_seqs:
            print(
                f"Update {len(update_seqs)} sequences, "
                f"nearly {len(update_seqs) / 450:.3f} hour"
            )
            results = get_keywords(
                update_seqs,
                workers_per_scan,
                ipr_scan_ex,
            )
            cache.update(results)
            with open(ipr_cache_path, "w") as f:
                json.dump(cache, f)
        else:
            print("No new sequences to update")

    print("<<< Finished loading <<<")
    return cache


def keyword_score_evaluate_worker(
    queue: mp.Queue,
    pid: int,
    subset: list[dict],
    design_batch_size: int,
    ipr_scan_ex: str,
    ipr_cache_path: str,
    workers_per_scan: int,
):
    if (
        design_batch_size is None
        or ipr_scan_ex is None
        or workers_per_scan is None
    ):
        raise ValueError(
            "Invalid kwargs: \n"
            f"design_batch_size: {design_batch_size}\n"
            f"interpro_scan_ex_path: {ipr_scan_ex}\n"
            f"workers_per_scan: {workers_per_scan}"
        )

    seq2keyw = load_dict(
        ipr_scan_ex=ipr_scan_ex,
        workers_per_scan=workers_per_scan,
        ipr_cache_path=ipr_cache_path,
        update_seqs=[item["reference"] for item in subset]
        + [
            item[f"response#{b}"]
            for item in subset
            for b in range(1, design_batch_size + 1)
        ],
    )

    results: list = [dict() for _ in range(len(subset))]
    for idx, item in enumerate(
        tqdm(
            subset,
            desc="IPRScore",
            ncols=100,
            disable=pid != 0,
        )
    ):
        ref_ids = list(
            set([item["InterPro-ID"] for item in seq2keyw[item["reference"]]])
        )
        res = {
            "instruction": item["instruction"],
            "reference": item["reference"],
            "reference_ipr_ids": ref_ids,
            **{
                f"response#{b}": item[f"response#{b}"]
                for b in range(1, design_batch_size + 1)
            },
        }
        for b in range(1, design_batch_size + 1):
            try:
                response = item[f"response#{b}"]
                res_ids = list(
                    set([item["InterPro-ID"] for item in seq2keyw[response]])
                )

                if len(ref_ids) == 0:
                    rec = float("nan")
                else:
                    rec = sum([id in set(ref_ids) for id in res_ids]) / len(
                        ref_ids
                    )
            except Exception:
                rec = float("nan")
                res_ids = [] if res_ids is None else []

            res.update(
                {
                    f"IPRRecovery#{b}": rec,
                    f"response_ipr_ids#{b}": res_ids,
                }
            )

        results[idx].update(res)

    queue.put((pid, results))


class IPRScoreMetric(BaseMetric):
    def __init__(self, config):
        super().__init__(config)
        self._name = config.ipr_score.name

    @property
    def metrics(self) -> list[str]:
        return ["IPRRecovery"]

    def summary(self, results: DataFrame) -> dict:
        bs = self.design_batch_size
        if bs == 1:
            return {
                "IPRRecovery": results["IPRRecovery#1"].mean() * 100,
            }
        else:
            ipr_recs = [
                results[f"IPRRecovery#{b}"].mean() * 100
                for b in range(1, bs + 1)
            ]
            return {
                "IPRRecovery": np.mean(ipr_recs),
                **{
                    f"IPRRecovery#{b}": ipr_recs[b - 1]
                    for b in range(1, bs + 1)
                },
            }


class IPRScoreEvaluator(BaseEvaluator):
    def __init__(self, config):
        super().__init__(config)
        self._name = config.ipr_score.name
        self.ipr_scan_ex = config.ipr_score.interpro_scan_ex_path
        self.workers_per_scan = config.ipr_score.workers_per_scan
        self.ipr_cache_path = config.ipr_score.interpro_cache_path

    def _execute_manual_multiprocess(self) -> None:
        results = multiprocess_evaluate(
            dataset=self.dataset,
            eval_worker=keyword_score_evaluate_worker,
            num_workers=self.num_cpu // self.workers_per_scan,
            kwargs={
                "design_batch_size": self.design_batch_size,
                "ipr_scan_ex": self.ipr_scan_ex,
                "ipr_cache_path": self.ipr_cache_path,
                "workers_per_scan": self.workers_per_scan,
            },
        )
        self.to_json(results)

    def execute(self) -> None:
        self._execute_manual_multiprocess()
