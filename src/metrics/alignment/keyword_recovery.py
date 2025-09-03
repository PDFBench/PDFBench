import json
import multiprocessing as mp
import os
import subprocess
import tempfile
import warnings

from src.metrics import BaseEvaluator, BaseMetric
from src.utils.multiprocess import multiprocess_evaluate


def get_keywords(
    sequences: list[str],
    num_workers: int,
    interproscan_path: str,
) -> dict[str, list[str]]:
    """
    _summary_

    :param List[str] sequences: _description_
    :param int num_workers: _description_
    :param str interproscan_path: _description_
    :param str temp_folder: _description_
    :return Dict[str, List[str]]: _description_
    """
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
            f"{num_workers}",
            "-T",
            temp_temp_folder,
            "-vl",
            "1",
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
    references: list[str],
    responses: list[str],
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


def keyword_recovery_evaluate_worker(
    queue: mp.Queue,
    pid: int,
    subset: list[dict],
    design_batch_size: int,
    interpro_scan_ex_path: str,
    workers_per_scan: int,
):
    if (
        design_batch_size is None
        or interpro_scan_ex_path is None
        or workers_per_scan is None
    ):
        raise ValueError(
            "Invalid kwargs: \n"
            f"design_batch_size: {design_batch_size}\n"
            f"interpro_scan_ex_path: {interpro_scan_ex_path}\n"
            f"workers_per_scan: {workers_per_scan}"
        )

    responses = [item["response"] for item in subset]
    references = [item["reference"] for item in subset]

    results = compute_keyword_recovery(
        references=references,
        responses=responses,
        num_workers=workers_per_scan,
        interpro_scan_path=interpro_scan_ex_path,
    )

    queue.put((pid, results))


class KeywordRecoveryMetric(BaseMetric):
    def __init__(self, config):
        super().__init__(config)
        self._name = config.protrek_score.name

    @property
    def metrics(self) -> list[str]:
        return ["KeywordRecovery"]


class KeywordRecoveryEvaluator(BaseEvaluator):
    def __init__(self, config):
        super().__init__(config)
        self._name = config.keyword_rec.name
        self.interpro_scan_ex_path = config.keyword_rec.interpro_scan_ex_path
        self.workers_per_scan = config.keyword_rec.workers_per_scan

    def _execute_manual_multiprocess(self) -> None:
        results = multiprocess_evaluate(
            dataset=self.dataset,
            eval_worker=keyword_recovery_evaluate_worker,
            num_workers=self.num_cpu // self.workers_per_scan,
            kwargs={
                "design_batch_size": self.design_batch_size,
                "interpro_scan_ex_path": self.interpro_scan_ex_path,
                "workers_per_scan": self.workers_per_scan,
            },
        )
        self.to_json(results)

    def execute(self) -> None:
        self._execute_manual_multiprocess()
