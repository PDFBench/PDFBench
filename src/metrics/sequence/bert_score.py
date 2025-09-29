import multiprocessing as mp
import warnings

import numpy as np
import torch
from accelerate.utils import gather_object
from pandas import DataFrame
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import EsmModel, EsmTokenizer, logging

from src.configs.sequence_args import BertModel
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


def compute_bertscore(
    pred_seq: str, ref_seq: str, model, tokenizer
) -> tuple[float, float, float]:
    """
    compute BertScore
    :param pred_seq: sequence predicted by model
    :param ref_seq: sequence ground truth
    :param model: model used to calculate BertScore
    :param tokenizer: tokenizor used by model
    :return: bert_f1, bert_precision, bert_recall
    """

    def get_embeddings(sequence):
        tokens = tokenizer(
            sequence, return_tensors="pt", truncation=True, max_length=512
        ).to(model.device)
        with torch.no_grad():
            outputs = model(**tokens)
        embeddings = outputs.last_hidden_state[0][1:-1]

        return embeddings

    pred_embed = get_embeddings(pred_seq)
    ref_embed = get_embeddings(ref_seq)
    if pred_embed.size(0) == 0 or ref_embed.size(0) == 0:
        return 0.0, 0.0, 0.0

    similarity_matrix = torch.cosine_similarity(
        pred_embed.unsqueeze(1), ref_embed.unsqueeze(0), dim=-1
    )

    precision = similarity_matrix.max(dim=1)[0].mean().item()
    recall = similarity_matrix.max(dim=0)[0].mean().item()
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

    return f1, precision, recall


def bertscore_evaluate_worker(
    queue: mp.Queue,
    pid: int,
    subset: list,
    **kwargs,
) -> None:
    design_batch_size = kwargs.get("design_batch_size")
    esm2_name_or_path = kwargs.get("esm2_name_or_path")
    esm2_batch_size = kwargs.get("esm2_batch_size")
    if (
        design_batch_size is None
        or esm2_name_or_path is None
        or esm2_batch_size is None
    ):
        raise ValueError(
            "Invalid kwargs: \n"
            f"design_batch_size: {design_batch_size}\n"
            f"esm2_name_or_path: {esm2_name_or_path}\n"
            f"esm2_batch_size: {esm2_batch_size}"
        )

    results: list = [dict() for _ in range(len(subset))]

    # region Bertscore based on ESM-2
    tokenizer = EsmTokenizer.from_pretrained(esm2_name_or_path)
    model = EsmModel.from_pretrained(esm2_name_or_path).to(f"cuda:{pid}")  # type: ignore
    model.eval()
    for idx, item in enumerate(
        tqdm(
            subset,
            desc="BertScore",
            ncols=100,
            disable=pid != 0,
        )
    ):
        res = {
            "instruction": item["instruction"],
            "reference": item["reference"],
        }
        for b in range(1, design_batch_size + 1):
            bert_f1, bert_precision, bert_recall = compute_bertscore(
                pred_seq=item[f"response#{b}"],
                ref_seq=item["reference"],
                model=model,
                tokenizer=tokenizer,
            )
            res.update(
                {
                    f"response#{b}": item[f"response#{b}"],
                    f"ESM2-F1#{b}": bert_f1,
                    f"ESM2-Precision#{b}": bert_precision,
                    f"ESM2-Recall#{b}": bert_recall,
                }
            )

        results[idx].update(res)
    # endregion

    queue.put((pid, results))


class BertScoreMetric(BaseMetric):
    def __init__(self, config):
        super().__init__(config)
        self.compute_models = config.bert_score.compute_models
        self._name = config.bert_score.name
        self.esm2_name_or_path = config.bert_score.esm2_name_or_path
        self.esm2_batch_size = config.bert_score.esm2_batch_size

    @property
    def metrics(self) -> list[str]:
        _metrics = []
        for model in BertModel:
            if model.name in self.compute_models:
                _metrics.extend(
                    [
                        f"{model.name}-F1",
                        f"{model.name}-Precision",
                        f"{model.name}-Recall",
                    ]
                )
        return _metrics

    def summary(self, results: DataFrame) -> dict:
        bs = self.design_batch_size
        if bs == 1:
            out = {}
            for model in BertModel:
                if model.name in self.compute_models:
                    out.update(
                        {
                            f"{model.name}-F1": results[
                                f"{model.name}-F1#1"
                            ].mean()
                            * 100,
                            f"{model.name}-Precision": results[
                                f"{model.name}-Precision#1"
                            ].mean()
                            * 100,
                            f"{model.name}-Recall": results[
                                f"{model.name}-Recall#1"
                            ].mean()
                            * 100,
                        }
                    )
        else:
            scores = {
                f"{model.name}": {
                    "F1": [
                        results[f"{model.name}-F1#{b}"].mean() * 100
                        for b in range(1, bs + 1)
                    ],
                    "Precision": [
                        results[f"{model.name}-Precision#{b}"].mean() * 100
                        for b in range(1, bs + 1)
                    ],
                    "Recall": [
                        results[f"{model.name}-Recall#{b}"].mean() * 100
                        for b in range(1, bs + 1)
                    ],
                }
                for model in BertModel
                if model.name in self.compute_models
            }

            out = {}
            for model in BertModel:
                if model.name in self.compute_models:
                    for label in ["F1", "Precision", "Recall"]:
                        out[f"{model.name}-{label}"] = (
                            rf"{np.mean(scores[model.name][label]):.2f}"
                            r"\(\pm\)"
                            rf"{np.std(scores[model.name][label], ddof=1):.2f}"
                        )
                        out.update(
                            {
                                f"{model.name}-{label}#{b}": scores[
                                    f"{model.name}"
                                ][label][b - 1]
                                for b in range(1, bs + 1)
                            }
                        )
        return out


class BertScoreEvaluator(BaseEvaluator):
    def __init__(self, config):
        super().__init__(config)
        self.compute_models = config.bert_score.compute_models
        self._name = config.bert_score.name
        self.esm2_name_or_path = config.bert_score.esm2_name_or_path
        self.esm2_batch_size = config.bert_score.esm2_batch_size

    def _execute_accelerate(self) -> None:
        dataloader = DataLoader(
            dataset=self.dataset,
            batch_size=self.esm2_batch_size,
            shuffle=False,
            drop_last=False,
        )

        MODEL_NAME = "EMS2"  # TODO: Support more models
        tokenizer = EsmTokenizer.from_pretrained(self.esm2_name_or_path)
        model = EsmModel.from_pretrained(self.esm2_name_or_path)

        # accelerate prepare
        model, dataloader = self.accelerator.prepare(model, dataloader)
        model.eval()
        all_results: list[dict] = []
        for batch in tqdm(
            dataloader,
            desc=f"BertScore # {self.accelerator.process_index}",
            postfix=f"Batch Size: {self.esm2_batch_size}",
            position=self.accelerator.process_index,
            # disable=not self.accelerator.is_main_process,
            ncols=120,
        ):
            batch_size = len(batch["instruction"])
            batch_results: list[dict] = []
            for i in range(batch_size):
                result_item = {
                    "instruction": batch["instruction"][i],
                    "reference": batch["reference"][i],
                }
                for b in range(1, self.design_batch_size + 1):
                    f1, precision, recall = compute_bertscore(
                        pred_seq=batch[f"response#{b}"][i],
                        ref_seq=batch["reference"][i],
                        model=model,
                        tokenizer=tokenizer,
                    )
                    result_item.update(
                        {
                            f"response#{b}": batch[f"response#{b}"][i],
                            f"{MODEL_NAME}-F1": f1,
                            f"{MODEL_NAME}-Precision": precision,
                            f"{MODEL_NAME}-Recall": recall,
                        }
                    )
                batch_results.append(result_item)

            all_results.extend(batch_results)

        print("All results: ", len(all_results))
        gathered_results: list[dict] = gather_object(all_results)
        print("Final results: ", len(gathered_results))

        if self.accelerator.is_main_process:
            self.to_json(gathered_results)

    def _excete_manual_multiprocess(self) -> None:
        results = multiprocess_evaluate(
            dataset=self.dataset,
            eval_worker=bertscore_evaluate_worker,
            num_workers=self.num_gpu,
            kwargs={
                "design_batch_size": self.design_batch_size,
                "esm2_name_or_path": self.esm2_name_or_path,
                "esm2_batch_size": self.esm2_batch_size,
            },
        )
        self.to_json(results)

    def execute(self) -> None:
        if self.speed_up:
            self._execute_accelerate()
        else:
            self._excete_manual_multiprocess()
