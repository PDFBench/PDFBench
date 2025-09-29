import math
import multiprocessing as mp
import warnings
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from accelerate.utils import gather_object
from esm.sdk.api import (
    ESMProtein,
    LogitsConfig,
)
from esm.tokenization import EsmSequenceTokenizer
from esm.utils.sampling import _BatchedESMProteinTensor
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    logging,
)

from src.configs.sequence_args import PerplexityModel
from src.datasets import BaseDataset
from src.metrics import BaseEvaluator, BaseMetric
from src.utils.context_manager import suppress_stdout
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


# region compute functions
def compute_perplexity_proglm_batch(sequences: list[str], tokenizer, model):
    sequences = [f"<gmask><sop><eos>{seq}" for seq in sequences]

    encodings = tokenizer(
        sequences, return_tensors="pt", padding=True, truncation=True
    ).to(model.device)

    input_ids = encodings["input_ids"]
    attention_mask = encodings["attention_mask"]

    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=input_ids)
        # logits: [batch, seq_len, vocab_size]
        logits = outputs.logits

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    shift_attention = attention_mask[:, 1:].contiguous()

    loss_fct = CrossEntropyLoss(reduction="none")
    token_losses = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
    ).view(shift_labels.size())

    token_losses = token_losses * shift_attention
    seq_loss = token_losses.sum(dim=1) / shift_attention.sum(dim=1)
    perplexities = [math.exp(loss.item()) for loss in seq_loss]

    return perplexities


def compute_perplexity_progen2(sequence: str, tokenizer, model) -> float:
    """
    Compute the perplexity of the protein sequence.
    :param sequence: sequence
    """
    input_ids = tokenizer(
        sequence,
        return_tensors="pt",
        max_length=1024,
        truncation=True,
    )["input_ids"].to(model.device)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)

    return math.exp(outputs.loss.item())


def compute_perplexity_rita(sequence: str, tokenizer, model) -> float:
    input_ids = tokenizer(
        sequence,
        return_tensors="pt",
        max_length=1024,
        truncation=True,
    )["input_ids"].to(model.device)
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            labels=input_ids,
        )
    return math.exp(outputs.loss.item())


def compute_perplexity_protgpt2(sequence: str, tokenizer, model) -> float:
    """
    Compute the perplexity of the protein sequence.
    :param sequence: sequence
    """
    sequence = sequence[: min(1024, len(sequence))]
    sequence = "\n".join(
        [sequence[beg : beg + 60] for beg in range(0, len(sequence), 60)]
    )
    sequence = "<|endoftext|>\n" + sequence + "\n<|endoftext|>"

    input_ids = torch.tensor(
        tokenizer.encode(sequence, max_length=1024, truncation=True)
    ).unsqueeze(0)
    input_ids = input_ids.to(model.device)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    loss, logits = outputs[:2]
    return math.exp(loss)


def compute_perplexity_proglm(
    sequence: str, tokenizer, model, prefix=False
) -> float:
    if prefix:
        sequence = f"<gmask><sop><eos>{sequence}"
    input_ids = tokenizer(
        sequence,
        return_tensors="pt",
    )["input_ids"].to(model.device)
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            labels=input_ids,
        )
    return math.exp(outputs.loss.item())


def compute_perplexity_esmc_batch(sequence: str, client, batch_size: int):
    # orginal input_ids
    mask_idx = EsmSequenceTokenizer().mask_token_id  # mask token id
    protein = ESMProtein(sequence=sequence)
    original_protein_tensor = client.encode(protein)
    original_token_ids = original_protein_tensor.sequence  # input_ids

    # masked input_ids
    positions_to_mask = list(range(1, original_token_ids.shape[0] - 1))
    masked_tensors = []
    for i in positions_to_mask:
        masked = original_token_ids.clone()
        masked[i] = mask_idx
        masked_tensors.append(masked)

    # calc
    log_probs = []
    for i in range(0, len(masked_tensors), batch_size):
        batch_sequences = masked_tensors[i : i + batch_size]
        batch_tensor = torch.stack(batch_sequences, dim=0).to(client.device)
        batched = _BatchedESMProteinTensor(
            sequence=batch_tensor,
        )
        with torch.no_grad():
            logits_output = client.logits(batched, LogitsConfig(sequence=True))

        # 对应位置的 token log-probs
        for j, masked_pos in enumerate(positions_to_mask[i : i + batch_size]):
            logits = logits_output.logits.sequence[j, masked_pos, :]
            log_probs_j = F.log_softmax(logits, dim=-1)
            token_log_prob = log_probs_j[original_token_ids[masked_pos]]
            log_probs.append(token_log_prob.item())

    ppl = float(np.exp(-np.mean(log_probs)))
    return ppl


# endregion compute functions


def perplexity_evaluate_worker(
    queue: mp.Queue,
    pid: int,
    subset: list,
    **kwargs,
):
    design_batch_size: int | None = kwargs.get("design_batch_size")
    batch_size: int | None = kwargs.get("batch_size")
    compute_models: list[PerplexityModel] | None = kwargs.get("compute_models")
    model2name_or_path: dict[PerplexityModel, str] | None
    model2name_or_path = kwargs.get("model2name_or_path")
    model2func: dict[PerplexityModel, Callable] | None
    model2func = kwargs.get("model2func")
    if (
        design_batch_size is None
        or batch_size is None
        or compute_models is None
        or model2name_or_path is None
        or model2func is None
    ):
        raise ValueError(
            "Invalid kwargs: \n"
            f"design_batch_size: {design_batch_size}\n"
            f"batch_size: {batch_size}\n"
            f"compute_models: {compute_models}\n"
            f"model2name_or_path: {model2name_or_path}\n"
            f"model2func: {model2func}"
        )

    results: list = [dict() for _ in range(len(subset))]
    for compute_model in PerplexityModel:
        if compute_model.name in compute_models:
            tokenizer = AutoTokenizer.from_pretrained(
                model2name_or_path[compute_model],
                trust_remote_code=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model2name_or_path[compute_model],
                trust_remote_code=True,
                torch_dtype="auto",
            ).to(f"cuda:{pid}")
            model.eval()

            for idx, item in enumerate(
                tqdm(
                    subset,
                    desc=f"Perplextiy - {compute_model.name}",
                    # postfix="Batch Size: Batch mode not supported",
                    disable=pid != 0,
                    ncols=120,
                )
            ):
                # init instruction, reference and responses
                res = {
                    "instruction": item["instruction"],
                    "reference": item["reference"],
                    **{
                        f"response#{b}": item[f"response#{b}"]
                        for b in range(1, design_batch_size + 1)
                    },
                }
                for b in range(1, design_batch_size + 1):
                    try:
                        ppl = model2func[compute_model](
                            sequence=item[f"response#{b}"],
                            tokenizer=tokenizer,
                            model=model,
                        )
                    except Exception:
                        ppl = float("nan")
                    res.update(
                        {
                            f"response#{b}": item[f"response#{b}"],
                            f"PPL-{compute_model.name}#{b}": ppl,
                        }
                    )

                if {"instruction", "reference"}.issubset(
                    results[idx].keys()
                ) and (
                    results[idx]["instruction"] != item["instruction"]
                    or results[idx]["reference"] != item["reference"]
                ):
                    raise RuntimeError(
                        "Error in Perplextiy Match: \n "
                        f"{item['instruction']} != {results[idx]['instruction']} \n"
                        f"{item['reference']} != {results[idx]['reference']}"
                    )

                results[idx].update(res)

    queue.put((pid, results))


class PerplexityMetric(BaseMetric):
    def __init__(self, config):
        super().__init__(config)
        self.compute_models = config.perplexity.compute_models
        self._name = config.perplexity.name
        self.batch_size = config.perplexity.batch_size
        self.model2name_or_path: dict[PerplexityModel, str] = {
            PerplexityModel.ProGen2: config.perplexity.progen2_name_or_path,
            PerplexityModel.ProtGPT2: config.perplexity.protgpt2_name_or_path,
            PerplexityModel.RITA: config.perplexity.rita_name_or_path,
            PerplexityModel.ProteinGLM: config.perplexity.proteinglm_name_or_path,
        }
        self.model2func: dict[PerplexityModel, Callable] = {
            PerplexityModel.ProGen2: compute_perplexity_progen2,
            PerplexityModel.ProtGPT2: compute_perplexity_protgpt2,
            PerplexityModel.RITA: compute_perplexity_rita,
            PerplexityModel.ProteinGLM: compute_perplexity_proglm,
        }

    @property
    def metrics(self) -> list[str]:
        _metrics = []
        for model in PerplexityModel:
            if model.name in self.compute_models:
                _metrics.append(f"PPL-{model.name}")
        return _metrics

    def summary(self, results) -> dict:
        bs = self.design_batch_size
        if bs == 1:
            return {
                f"PPL-{model.name}": results[f"PPL-{model.name}#1"].mean()
                for model in PerplexityModel
                if model.name in self.compute_models
                and model != PerplexityModel.ProteinGLM
            }
        else:
            ppls = {
                f"PPL-{model.name}": [
                    results[f"PPL-{model.name}#{b}"].mean()
                    for b in range(1, bs + 1)
                ]
                for model in PerplexityModel
                if model.name in self.compute_models
                and model != PerplexityModel.ProteinGLM
            }

            out = {}
            for model in PerplexityModel:
                if model == PerplexityModel.ProteinGLM:
                    continue
                if model.name in self.compute_models:
                    out[f"PPL-{model.name}"] = (
                        rf"{np.mean(ppls[f'PPL-{model.name}']):.2f}"
                        r"\(\pm\)"
                        rf"{np.std(ppls[f'PPL-{model.name}'], ddof=1):.2f}"
                    )

                    out.update(
                        {
                            f"PPL-{model.name}#{b}": ppls[f"PPL-{model.name}"][
                                b - 1
                            ]
                            for b in range(1, bs + 1)
                        }
                    )
            return out


class PerplexityEvaluator(BaseEvaluator):
    def __init__(self, config):
        super().__init__(config)
        self.compute_models = config.perplexity.compute_models
        self._name = config.perplexity.name
        self.batch_size = config.perplexity.batch_size
        self.model2name_or_path: dict[PerplexityModel, str] = {
            PerplexityModel.ProGen2: config.perplexity.progen2_name_or_path,
            PerplexityModel.ProtGPT2: config.perplexity.protgpt2_name_or_path,
            PerplexityModel.RITA: config.perplexity.rita_name_or_path,
            PerplexityModel.ProteinGLM: config.perplexity.proteinglm_name_or_path,
        }
        self.model2func: dict[PerplexityModel, Callable] = {
            PerplexityModel.ProGen2: compute_perplexity_progen2,
            PerplexityModel.ProtGPT2: compute_perplexity_protgpt2,
            PerplexityModel.RITA: compute_perplexity_rita,
            PerplexityModel.ProteinGLM: compute_perplexity_proglm,
        }

    def _evaluate_accelerate_single(
        self, dataset: BaseDataset, compute_model: PerplexityModel
    ) -> list[dict[str, float | str]] | None:
        with suppress_stdout():
            tokenizer = AutoTokenizer.from_pretrained(
                self.model2name_or_path[compute_model],
                trust_remote_code=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                self.model2name_or_path[compute_model],
                trust_remote_code=True,
                torch_dtype="auto",
            )
        model.eval()
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )

        model, dataloader = self.accelerator.prepare(model, dataloader)

        all_results: list[dict] = []
        for batch in tqdm(
            dataloader,
            desc=f"Perplextiy - {compute_model.name}",
            postfix=f"Batch Size: {self.batch_size}",
            disable=not self.accelerator.is_main_process,
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
                    ppl = self.model2func[compute_model](
                        sequence=batch[f"response#{b}"][i],
                        tokenizer=tokenizer,
                        model=model,
                    )
                    result_item.update(
                        {
                            f"response#{b}": batch[f"response#{b}"][i],
                            f"PPL-{compute_model.name}#{b}": ppl,
                        }
                    )
                batch_results.append(result_item)

            all_results.extend(batch_results)

        gathered_results: list[dict] = gather_object(all_results)

        del model, tokenizer
        torch.cuda.empty_cache()
        if self.accelerator.is_main_process:
            return gathered_results

    def _execute_accelerate(self) -> list[dict]:  # type: ignore
        # TODO: Support Accelerate
        # Compute PPLs using different models
        results: list[dict] = []
        for model in PerplexityModel:
            if model.name in self.compute_models:
                results.append(
                    self._evaluate_accelerate_single(
                        self.dataset,
                        model,
                    )  # type: ignore
                )

        # Conbine the results of PPLs
        final_results: list[dict] = []
        for idx in range(len(self.dataset)):
            corr_inst = self.dataset[idx]["instruction"]  # type: ignore
            corr_ref = self.dataset[idx]["reference"]  # type: ignore

            tmp = {}
            for idy in range(len(results)):
                res = results[idy][idx]
                if (
                    res["instruction"] != corr_inst
                    or res["reference"] != corr_ref
                ):
                    raise RuntimeError(
                        "Error in Perplextiy: \n "
                        f"{corr_inst} != {res['instruction']} \n"
                        f"{corr_ref} != {res['reference']}"
                    )
                tmp.update(res)

            final_results.append(tmp)

        if self.accelerator.is_main_process:
            self.to_json(final_results)

    def _execute_manual_multiprocess(self) -> None:
        results = multiprocess_evaluate(
            dataset=self.dataset,
            eval_worker=perplexity_evaluate_worker,
            num_workers=self.num_gpu,
            kwargs={
                "design_batch_size": self.design_batch_size,
                "batch_size": self.batch_size,
                "compute_models": self.compute_models,
                "model2name_or_path": self.model2name_or_path,
                "model2func": self.model2func,
            },
        )
        self.to_json(results)

    def execute(self) -> None:
        if self.speed_up:
            self._execute_accelerate()
        else:
            self._execute_manual_multiprocess()
