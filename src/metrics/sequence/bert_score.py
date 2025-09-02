import json
import warnings

import torch
from accelerate.utils import gather_object
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import EsmModel, EsmTokenizer, logging

from src.configs.sequence_args import BertModel
from src.metrics import BaseEvaluator, BaseMetric

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


class BertScoreMetric(BaseMetric):
    def __init__(self, config):
        super().__init__(config)
        self.compute_models = config.bert_score.compute_models
        self._name = config.bert_score.name
        self._speed_up = config.bert_score.speed_up
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


class BertScoreEvaluator(BaseEvaluator):
    def __init__(self, config):
        super().__init__(config)
        self.compute_models = config.bert_score.compute_models
        self._name = config.bert_score.name
        self.esm2_name_or_path = config.bert_score.esm2_name_or_path
        self.esm2_batch_size = config.bert_score.esm2_batch_size

    def execute(self) -> None:
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
        # endregion

        if self.accelerator.is_main_process:
            print(len(gathered_results))
            with open(self.output_path, "w") as f:
                json.dump(gathered_results, f)
