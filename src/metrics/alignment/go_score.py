import math
import multiprocessing as mp
import os
import tempfile

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from src.utils import logging
from src.utils.context_manager import suppress_stdout
from src.utils.multiprocess import multiprocess_evaluate

from ..metric import BaseEvaluator, BaseMetric

logger = logging.get_logger(__name__)


def deepgo_predict(
    sequences: list[str],
    deepgo_weight_path: str,
    threshold: float = 0.1,
    batch_size: int = 32,
    device: str = "cpu",
    use_bp: bool = False,
    use_mf: bool = False,
    use_cc: bool = False,
):
    from .models.DeepGO2.data import load_normal_forms
    from .models.DeepGO2.extract_esm import extract_esm
    from .models.DeepGO2.models import DeepGOModel
    from .models.DeepGO2.utils import Ontology

    with tempfile.TemporaryDirectory() as temp_folder:
        tmp_fasta_path = os.path.join(temp_folder, "temp_input.fasta")
        with open(tmp_fasta_path, "w") as f:
            for i, seq in enumerate(sequences):
                f.write(f">protein_{i}\n{seq}\n")

        logger.debug("Extracting ESM Embedding")
        with suppress_stdout():
            proteins, data = extract_esm(
                tmp_fasta_path, out_file=None, device=device
            )
        proteins = sequences

        onts = []
        if use_bp:
            onts.append("bp")
        if use_mf:
            onts.append("mf")
        if use_cc:
            onts.append("cc")

        go_file = f"{deepgo_weight_path}/go.obo"
        go_norm = f"{deepgo_weight_path}/go-plus.norm"
        go = Ontology(go_file, with_rels=True)

        ent_models = {
            "mf": [0, 1, 2, 5, 6, 8],
            "bp": [2, 5, 6, 7, 8, 9],
            "cc": [1, 3, 4, 5, 6, 7],
        }

        results = {p: {ont: [] for ont in onts} for p in proteins}
        for ont in onts:
            terms_file = f"{deepgo_weight_path}/{ont}/terms.pkl"
            terms_df = pd.read_pickle(terms_file)
            terms = terms_df["gos"].values.flatten()  # type: ignore
            terms_dict = {v: i for i, v in enumerate(terms)}

            n_terms = len(terms_dict)

            nf1, nf2, nf3, nf4, relations, zero_classes = load_normal_forms(
                go_norm, terms_dict
            )
            n_rels = len(relations)
            n_zeros = len(zero_classes)

            sum_preds = np.zeros((len(proteins), n_terms), dtype=np.float32)
            model = DeepGOModel(2560, n_terms, n_zeros, n_rels, device).to(
                device
            )

            for mn in ent_models[ont]:
                model_file = (
                    f"{deepgo_weight_path}/{ont}/deepgozero_esm_plus_{mn}.th"
                )
                model.load_state_dict(
                    torch.load(model_file, map_location=device)
                )
                model.eval()

                with torch.no_grad():
                    steps = int(math.ceil(len(proteins) / batch_size))
                    preds = []
                    for i in range(steps):
                        start, end = i * batch_size, (i + 1) * batch_size
                        batch_features = data[start:end].to(device)
                        logits = model(batch_features)
                        preds.append(logits.detach().cpu().numpy())
                    preds = np.concatenate(preds)
                sum_preds += preds

            preds = sum_preds / len(ent_models[ont])

            name2type = {
                "molecular_function": "mf",
                "biological_process": "bp",
                "cellular_component": "cc",
            }
            for i, prot in enumerate(proteins):
                above_threshold = np.argwhere(preds[i] >= threshold).flatten()
                for j in above_threshold:
                    namesapce = name2type[go.get_term(terms[j])["namespace"]]  # type: ignore
                    name = go.get_term(terms[j])["name"]  # type: ignore
                    results[prot][namesapce].append(
                        {
                            "GO-ID": terms[j],
                            "DeepGO-Score": float(preds[i, j]),
                            "GO-Name": name,
                            "GO-Type": namesapce,
                        }
                    )

        return results


def go_score_evaluate_worker(
    queue: mp.Queue,
    pid: int,
    subset: list[dict],
    design_batch_size: int,
    deepgo_weight_path: str,
    deepgo_threshold: float,
    deepgo_batch_size: int,
):
    seq2go = deepgo_predict(
        [
            item[f"response#{b}"]
            for item in subset
            for b in range(1, design_batch_size + 1)
        ]
        + [item["reference"] for item in subset],
        deepgo_weight_path=deepgo_weight_path,
        threshold=deepgo_threshold,
        batch_size=deepgo_batch_size,
        device=f"cuda:{pid}",
        use_mf=True,
    )

    results: list = [dict() for _ in range(len(subset))]
    for idx, item in enumerate(
        tqdm(
            subset,
            desc="GOScore",
            ncols=100,
            disable=pid != 0,
        )
    ):
        ref_ids = list(
            set([item["GO-ID"] for item in seq2go[item["reference"]]["mf"]])
        )
        res = {
            "instruction": item["instruction"],
            "reference": item["reference"],
            "reference_go_ids": ref_ids,
            **{
                f"response#{b}": item[f"response#{b}"]
                for b in range(1, design_batch_size + 1)
            },
        }
        for b in range(1, design_batch_size + 1):
            response = item[f"response#{b}"]
            res_ids = list(
                set([item["GO-ID"] for item in seq2go[response]["mf"]])
            )

            if len(ref_ids) == 0:
                rec = float("nan")
            else:
                rec = sum([id in set(ref_ids) for id in res_ids]) / len(ref_ids)

            res.update(
                {
                    f"GORecovery#{b}": rec,
                    f"response_go_ids#{b}": res_ids,
                }
            )

        results[idx].update(res)

    queue.put((pid, results))


class GOScoreMetric(BaseMetric):
    def __init__(self, config):
        super().__init__(config)
        self._name = config.go_score.name
        self.deepgo_threshold = config.go_score.deepgo_threshold
        self.deepgo_batch_size = config.go_score.deepgo_batch_size

    @property
    def metrics(self) -> list:
        return ["GOScore"]

    def summary(self, results: pd.DataFrame) -> dict:
        bs = self.design_batch_size
        if bs == 1:
            return {
                "GORecovery": results["GORecovery#1"].mean() * 100,
            }
        else:
            go_recs = [
                results[f"GORecovery#{b}"].mean() * 100
                for b in range(1, bs + 1)
            ]
            return {
                "GORecovery": (
                    rf"{np.mean(go_recs):.2f}"
                    r"\(\pm\)"
                    rf"{np.std(go_recs, ddof=1):.2f}"
                ),
                **{f"GORecovery#{b}": go_recs[b - 1] for b in range(1, bs + 1)},
            }


class GOScoreEvaluator(BaseEvaluator):
    def __init__(self, config):
        super().__init__(config)
        self._name = config.go_score.name
        self.deepgo_weight_path = config.go_score.deepgo_weight_path
        self.deepgo_threshold = config.go_score.deepgo_threshold
        self.deepgo_batch_size = config.go_score.deepgo_batch_size

    def _execute_manual_multiprocess(self) -> None:
        results = multiprocess_evaluate(
            dataset=self.dataset,
            eval_worker=go_score_evaluate_worker,
            num_workers=self.num_gpu,
            kwargs={
                "design_batch_size": self.design_batch_size,
                "deepgo_weight_path": self.deepgo_weight_path,
                "deepgo_threshold": self.deepgo_threshold,
                "deepgo_batch_size": self.deepgo_batch_size,
            },
        )
        self.to_json(results)

    def execute(self) -> None:
        self._execute_manual_multiprocess()
