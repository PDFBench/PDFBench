import multiprocessing as mp
import os
import warnings

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from transformers import BertModel, BertTokenizer, logging

from src.utils.context_manager import suppress_stdout
from src.utils.multiprocess import multiprocess_evaluate

from ..metric import BaseEvaluator, BaseMetric
from .models.EvoLlama.infer import infer, init_evo_llama

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
    message="`torch.cuda.amp.autocast(args...)`.*",
)


def get_embedding(
    model: BertModel, tokenizer: BertTokenizer, texts: list[str]
) -> torch.Tensor:
    def mean_pooling(output, mask):
        embeddings = output[
            0
        ]  # First element of model_output contains all token embeddings
        mask = mask.unsqueeze(-1).expand(embeddings.size()).float()
        return torch.sum(embeddings * mask, 1) / torch.clamp(
            mask.sum(1), min=1e-9
        )

    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)

    return mean_pooling(outputs, inputs["attention_mask"])


def evollama_score_evaluate_worker(
    queue: mp.Queue,
    pid: int,
    subset: list[dict],
    design_batch_size: int,
    evollama_path: str,
    llama_name_or_path: str,
    pubmedbert_name_or_path: str,
):
    if (
        design_batch_size is None
        or evollama_path is None
        or llama_name_or_path is None
        or pubmedbert_name_or_path is None
    ):
        raise ValueError(
            "Invalid kwargs: \n"
            f"design_batch_size: {design_batch_size}\n"
            f"evollama_path: {evollama_path}\n"
            f"llama_name_or_path: {llama_name_or_path}\n"
            f"pubmedbert_name_or_path: {pubmedbert_name_or_path}"
        )

    prompt = "The function of the protein is:\n"
    prompt = [prompt]
    with suppress_stdout():
        model = (
            init_evo_llama(
                structure_encoder_path=os.path.join(
                    evollama_path, "structure_encoder_weights.bin"
                ),
                structure_encoder_name="ProteinMPNN",
                sequence_encoder_path=os.path.join(
                    evollama_path, "sequence_encoder"
                ),
                llm_path=llama_name_or_path,
                projection_path=os.path.join(
                    evollama_path, "projection_weights.bin"
                ),
                projection_fusion=False,
                is_inference=True,
                llm_embedding_dim=3072,
            )
            .eval()
            .to(f"cuda:{pid}")  # type: ignore
        )

    embedding_model = BertModel.from_pretrained(
        pubmedbert_name_or_path,
    ).to(f"cuda:{pid}")  # type: ignore
    embedding_tokenizer = BertTokenizer.from_pretrained(
        pubmedbert_name_or_path,
    )

    results = []
    with torch.no_grad():
        for item in tqdm(
            subset,
            desc="Evollama Score",
            ncols=100,
            disable=pid != 0,
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
                evollama_inst = infer(
                    model,
                    None,  # type: ignore
                    [[item[f"response#{b}"]]],
                    prompt,
                )[0]
                embeddings = get_embedding(
                    embedding_model,  # type: ignore
                    embedding_tokenizer,
                    [evollama_inst, item["instruction"]],
                )
                res.update(
                    {
                        f"evollama_instruction#{b}": evollama_inst,
                        f"evollama_score#{b}": nn.functional.cosine_similarity(
                            embeddings[0], embeddings[1], dim=0
                        ).item(),
                    }
                )
            results.append(res)

    queue.put((pid, results))


class EvoLlamaScoreMetric(BaseMetric):
    def __init__(self, config):
        super().__init__(config)
        self._name = config.evollama_score.name

    @property
    def metrics(self) -> list[str]:
        return ["EvoLlamaScore"]

    def summary(self, results) -> dict[str, float]:
        bs = self.design_batch_size
        _summary = {}
        if bs == 1:
            _summary["EvoLlamaScore"] = results["evollama_score#1"].mean() * 100
        else:
            evollama_scores = [
                results[f"evollama_score#{b}"].mean() * 100
                for b in range(1, bs + 1)
            ]
            _summary["EvoLlamaScore"] = (
                rf"{np.mean(evollama_scores):.2f}"
                r"\(\pm\)"
                rf"{np.std(evollama_scores, ddof=1):.2f}"
            )
            _summary.update(
                {
                    f"evollama_score#{b}": evollama_scores[b - 1]
                    for b in range(1, bs + 1)
                }
            )
        return _summary


class EvoLlamaScoreEvaluator(BaseEvaluator):
    def __init__(self, config):
        super().__init__(config)
        self._name = config.evollama_score.name
        self.evollama_path = config.evollama_score.evollama_path
        self.llama_name_or_path = config.evollama_score.llama_name_or_path
        self.pubmedbert_name_or_path = (
            config.evollama_score.pubmedbert_name_or_path
        )

    def _execute_acclerate(self) -> None:
        raise NotImplementedError

    def _execute_manual_multiprocess(self) -> None:
        results = multiprocess_evaluate(
            dataset=self.dataset,
            eval_worker=evollama_score_evaluate_worker,
            num_workers=self.num_gpu,
            kwargs={
                "design_batch_size": self.design_batch_size,
                "evollama_path": self.evollama_path,
                "llama_name_or_path": self.llama_name_or_path,
                "pubmedbert_name_or_path": self.pubmedbert_name_or_path,
            },
        )
        self.to_json(results)

    def execute(self) -> None:
        if self.speed_up:
            self._execute_acclerate()
        else:
            self._execute_manual_multiprocess()
