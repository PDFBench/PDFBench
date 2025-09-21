import multiprocessing as mp
import os
import subprocess
import warnings

import numpy as np
from tqdm.auto import tqdm
from transformers import EsmForProteinFolding, EsmTokenizer, logging

from src.metrics import BaseEvaluator, BaseMetric
from src.utils.folding import seq_to_md5, seq_to_struc
from src.utils.multiprocess import multiprocess_evaluate

logging.set_verbosity_error()
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="TypedStorage is deprecated.*",
)
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message="`clean_up_tokenization_spaces` was not set.*",
)
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message="`torch.cuda.amp.autocast(args...)`",
)


def compute_tm_score(
    ref: str,
    res: str,
    tm_score_path: str,
    model: EsmForProteinFolding,
    tokenizer: EsmTokenizer,
    pdb_cache_dir: str,
):
    try:
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
        if not os.path.exists(pdb_res):
            seq_to_struc(
                tokenizer=tokenizer,
                model=model,
                sequences=[res],
                pdb_cache_dir=pdb_cache_dir,
                return_foldability=False,
            )
    except Exception as e:
        warnings.warn(f"TmScore Error with {e}")
        return float("nan")

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
    except Exception as e:
        warnings.warn(f"TmScore Error with {e}")
        return float("nan")


def tm_score_evaluate_worker(
    queue: mp.Queue,
    pid: int,
    subset: list,
    design_batch_size: int,
    tm_score_ex_path: str,
    esm_fold_name_or_path: str,
    pdb_cache_dir: str,
) -> None:
    if (
        tm_score_ex_path is None
        or esm_fold_name_or_path is None
        or pdb_cache_dir is None
    ):
        raise ValueError(
            "Invalid kwargs: \n"
            f"design_batch_size: {tm_score_ex_path}\n"
            f"esm_fold_name_or_path: {esm_fold_name_or_path}\n"
            f"pdb_cache_dir: {pdb_cache_dir}"
        )

    tokenizer = EsmTokenizer.from_pretrained(esm_fold_name_or_path)
    model = EsmForProteinFolding.from_pretrained(esm_fold_name_or_path).to(
        f"cuda:{pid}"  # type: ignore
    )
    model.esm = model.esm.float()
    model.trunk.set_chunk_size(64)  # type: ignore

    results: list = [dict() for _ in range(len(subset))]
    for idx, item in enumerate(
        tqdm(
            subset,
            desc="GT-TMScore",
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
        for b in range(1, design_batch_size + 1):
            res.update(
                {
                    f"GT-TMScore#{b}": compute_tm_score(
                        ref=item["reference"],
                        res=item[f"response#{b}"],
                        tm_score_path=tm_score_ex_path,
                        model=model,
                        tokenizer=tokenizer,
                        pdb_cache_dir=pdb_cache_dir,
                    )
                }
            )

        results[idx].update(res)

    queue.put((pid, results))


class TMScoreMetric(BaseMetric):
    def __init__(self, config):
        super().__init__(config)
        self._name = config.tm_score.name

    @property
    def metrics(self) -> list[str]:
        return ["GT-TMScore"]

    def summary(self, results) -> dict:
        bs = self.design_batch_size
        _summary = {}
        if bs == 1:
            _summary["GT-TMScore"] = results["GT-TMScore#1"].mean() * 100
        else:
            tm_scores = [
                results[f"GT-TMScore#{b}"].mean() * 100
                for b in range(1, bs + 1)
            ]
            _summary["GT-TMScore"] = (
                rf"{np.mean(tm_scores):.2f}"
                r"\(\pm\)"
                rf"{np.std(tm_scores, ddof=1):.2f}"
            )
            _summary.update(
                {f"GT-TMScore#{b}": tm_scores[b - 1] for b in range(1, bs + 1)}
            )
        return _summary


class TMScoreEvaluator(BaseEvaluator):
    def __init__(self, config):
        super().__init__(config)
        self._name = config.tm_score.name
        self.tm_score_ex_path = config.tm_score.tm_score_ex_path
        self.esm_fold_name_or_path = config.tm_score.esm_fold_name_or_path
        self.pdb_cache_dir = config.tm_score.pdb_cache_dir

    def execute(self):
        results = multiprocess_evaluate(
            dataset=self.dataset,
            eval_worker=tm_score_evaluate_worker,
            num_workers=self.num_gpu,
            kwargs={
                "design_batch_size": self.design_batch_size,
                "tm_score_ex_path": self.tm_score_ex_path,
                "esm_fold_name_or_path": self.esm_fold_name_or_path,
                "pdb_cache_dir": self.pdb_cache_dir,
            },
        )
        self.to_json(results)
