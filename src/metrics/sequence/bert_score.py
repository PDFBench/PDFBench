import torch
from transformers import logging

from src.configs.sequence_args import BertModel
from src.metrics import BaseEvaluator, BaseMetric

logging.set_verbosity_error()


def compute_bertscore(pred_seq: str, ref_seq: str, model, tokenizer) -> tuple:
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
        self.esm2_name_or_path = config.bert_score.esm2_name_or_path
        self.esm2_batch_size = config.bert_score.esm2_batch_size

    def execute(self) -> None:
        pass
