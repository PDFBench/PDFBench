import hashlib
import multiprocessing as mp
import os
import warnings

import biotite.structure.io as bsio
import numpy as np
import torch
from accelerate.utils import gather_object
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import EsmForProteinFolding, EsmTokenizer, logging

from src.metrics import BaseEvaluator, BaseMetric
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


def get_md5_sequence(sequence: str) -> str:
    return hashlib.md5(sequence.encode()).hexdigest()


def get_pae(output):
    pae = (
        output["aligned_confidence_probs"][0].cpu().numpy() * np.arange(64)
    ).mean(-1) * 31
    mask = output["atom37_atom_exists"][0, :, 1] == 1
    mask = mask.cpu()
    pae = pae[mask, :][:, mask]

    # PAE is a matrix with size of [L, L], representing global confidence across a sequence
    # Here we use the mean of the matrix as the global confidence score
    return pae.mean()


def compute_foldability(
    tokenizer,
    model: EsmForProteinFolding,
    sequences: list[str],
    pdb_cache_dir: str,
) -> list[dict]:
    input_ids = tokenizer(
        sequences,
        return_tensors="pt",
        padding=True,
        truncation=True,
        add_special_tokens=False,
        max_length=1024,
    )["input_ids"].to(model.device)
    pae_scores = []
    with (
        torch.no_grad()
    ):  # TODO: torch.cuda.empty_cache() and torch.inference_mode()
        output = model(input_ids)
        pae_scores.append(get_pae(output))

    pdbs = model.output_to_pdb(output)
    md5_sequences = [get_md5_sequence(seq) for seq in sequences]

    plddt_scores = []
    for md5_sequence, pdb in zip(md5_sequences, pdbs):
        save_path = os.path.join(pdb_cache_dir, f"{md5_sequence}.pdb")
        with open(save_path, "w") as f:
            f.write(pdb)

        struct = bsio.load_structure(save_path, extra_fields=["b_factor"])
        plddt_scores.append(struct.b_factor.mean())

    ret = []
    for sequence, md5_sequence, plddt, pae in zip(
        sequences, md5_sequences, plddt_scores, pae_scores
    ):
        ret.append(
            {
                "sequence": sequence,
                "pdb_file_name": f"{md5_sequence}.pdb",
                "pLDDT": plddt,
                "pAE": pae.mean(),
            }
        )
    return ret


def foldability_evaluate_worker(
    queue: mp.Queue,
    pid: int,
    subset: list,
    design_batch_size: int | None = None,
    pdb_cache_dir: str | None = None,
    esm_fold_name_or_path: str | None = None,
) -> None:
    if (
        design_batch_size is None
        or pdb_cache_dir is None
        or esm_fold_name_or_path is None
    ):
        raise ValueError(
            "Invalid kwargs: \n"
            f"design_batch_size: {design_batch_size}\n"
            f"pdb_cache_dir: {pdb_cache_dir}\n"
            f"esm_fold_name_or_path: {esm_fold_name_or_path}"
        )

    tokenizer = EsmTokenizer.from_pretrained(esm_fold_name_or_path)
    model = EsmForProteinFolding.from_pretrained(esm_fold_name_or_path).to(
        f"cuda:{pid}"  # type: ignore
    )
    model.esm = model.esm.float()
    model.trunk.set_chunk_size(64)

    results: list = [dict() for _ in range(len(subset))]
    idx = 0

    for idx, item in enumerate(
        tqdm(
            subset,
            desc="Foldability",
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
            res.update({f"response#{b}": item[f"response#{b}"]})

            try:
                tmp = compute_foldability(
                    tokenizer,
                    model,
                    [item[f"response#{b}"]],
                    pdb_cache_dir,
                )[0]
            except Exception:
                tmp = {
                    "pdb_file_name": "None",
                    "pLDDT": float("nan"),
                    "pAE": float("nan"),
                }
            pdb_file_name, plddt, pae = (
                tmp["pdb_file_name"],
                tmp["pLDDT"],
                tmp["pAE"],
            )
            res.update(
                {
                    f"pLDDT#{b}": plddt,
                    f"pAE#{b}": pae,
                    f"pdb_file_name#{b}": pdb_file_name,
                }
            )
        results[idx].update(res)

    queue.put((pid, results))


class FoldabilityMetric(BaseMetric):
    def __init__(self, config):
        super().__init__(config)
        self._name = config.foldability.name
        self.esm_fold_name_or_path = config.foldability.esm_fold_name_or_path
        self.pdb_cache_dir = config.foldability.pdb_cache_dir

    @property
    def metrics(self) -> list[str]:
        return ["pLDDT", "pLDDT>70", "pAE", "pAE<10"]

    def summary(self, results) -> dict:
        bs = self.design_batch_size
        if bs == 1:
            return {
                "pLDDT": results["pLDDT#1"].mean() * 100,
                "pLDDT>70": (results["pLDDT#1"] > 0.7).mean() * 100,
                "pAE": results["pAE#1"].mean(),
                "pAE<10": (results["pAE#1"] < 10).mean() * 100,
            }
        else:
            plddt = [
                results[f"pLDDT#{b}"].mean() * 100 for b in range(1, bs + 1)
            ]
            plddt_gt_70 = [
                (results[f"pLDDT#{b}"] > 0.7).mean() * 100
                for b in range(1, bs + 1)
            ]
            pae = [results[f"pAE#{b}"].mean() for b in range(1, bs + 1)]
            pae_lt_10 = [
                (results[f"pAE#{b}"] < 10).mean() * 100
                for b in range(1, bs + 1)
            ]
            return {
                "pLDDT": (  # pLDDT
                    rf"{np.mean(plddt):.2f}"
                    r"\(\pm\)"
                    rf"{np.std(plddt, ddof=1):.2f}"
                ),
                **{f"pLDDT#{b}": plddt[b - 1] for b in range(1, bs + 1)},
                "pLDDT>70": (  # pLDDT > 70
                    rf"{np.mean(plddt_gt_70):.2f}"
                    r"\(\pm\)"
                    rf"{np.std(plddt_gt_70, ddof=1):.2f}"
                ),
                **{
                    f"pLDDT>70#{b}": plddt_gt_70[b - 1]
                    for b in range(1, bs + 1)
                },
                "pAE": (  # pAE
                    rf"{np.mean(pae):.2f}"
                    r"\(\pm\)"
                    rf"{np.std(pae, ddof=1):.2f}"
                ),
                **{f"pAE#{b}": pae[b - 1] for b in range(1, bs + 1)},
                "pAE<10": (  # pAE < 10
                    rf"{np.mean(pae_lt_10):.2f}"
                    r"\(\pm\)"
                    rf"{np.std(pae_lt_10, ddof=1):.2f}"
                ),
                **{f"pAE<10#{b}": pae_lt_10[b - 1] for b in range(1, bs + 1)},
            }


class FoldabilityEvaluator(BaseEvaluator):
    def __init__(self, config):
        super().__init__(config)
        self._name = config.foldability.name
        self.pdb_cache_dir = os.path.join(
            self.output_dir, config.foldability.pdb_cache_dir
        )
        self.esm_fold_name_or_path = config.foldability.esm_fold_name_or_path

        os.makedirs(self.pdb_cache_dir, exist_ok=True)

    def _execute_acclerate(self) -> list[dict]:  # type: ignore
        # region ESM2-based BertScore
        tokenizer = EsmTokenizer.from_pretrained(self.esm_fold_name_or_path)
        model = EsmForProteinFolding.from_pretrained(self.esm_fold_name_or_path)
        model.esm = model.esm.float()
        model.trunk.set_chunk_size(64)
        dataloader = DataLoader(
            dataset=self.dataset,
            batch_size=1,  # TODO: support batch
            shuffle=False,
        )
        model, dataloader = self.accelerator.prepare(model, dataloader)

        all_results: list[dict] = []
        for batch in tqdm(
            dataloader,
            desc="Foldability",
            disable=not self.accelerator.is_main_process,
        ):
            batch_size = len(batch["instruction"])
            batch_results: list[dict] = []

            # TODO: support batch
            for i in range(batch_size):
                result_item = {
                    "instruction": batch["instruction"][i],
                    "reference": batch["reference"][i],
                }
                for b in range(1, self.design_batch_size + 1):
                    result_item.update(
                        {f"response#{b}": batch[f"response#{b}"][i]}
                    )
                    _, pdb_file_name, plddt, pae = compute_foldability(
                        tokenizer,
                        model,
                        [batch[f"response#{b}"][i]],
                        self.pdb_cache_dir,
                    )[0]
                    result_item.update(
                        {
                            f"pLDDT#{b}": plddt,
                            f"pAE#{b}": pae,
                            f"pdb_file_name#{b}": pdb_file_name,
                        }
                    )
                batch_results.append(result_item)

            all_results.extend(batch_results)

        gathered_results: list[dict] = gather_object(all_results)
        # endregion

        del model, tokenizer
        torch.cuda.empty_cache()
        if self.accelerator.is_main_process:
            return gathered_results

    def _execute_manual_multiprocess(self):
        results = multiprocess_evaluate(
            dataset=self.dataset,
            eval_worker=foldability_evaluate_worker,
            num_workers=self.num_gpu,
            kwargs={
                "design_batch_size": self.design_batch_size,
                "pdb_cache_dir": self.pdb_cache_dir,
                "esm_fold_name_or_path": self.esm_fold_name_or_path,
            },
        )
        self.to_json(results)

    def execute(self) -> None:
        if self.speed_up:
            self._execute_acclerate()
        else:
            self._execute_manual_multiprocess()
