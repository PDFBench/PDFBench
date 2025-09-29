# This file is copied from [EvoLlama](https://github.com/sornkL/EvoLlama)
# Original license: MIT License
from typing import List

import torch

from .core import EvoLlama, EvoLlamaConfig


def init_evo_llama(
    structure_encoder_path: str | None = None,
    structure_encoder_name: str | None = None,
    sequence_encoder_path: str | None = None,
    llm_path: str | None = None,
    projection_path: str | None = None,
    projection_fusion: bool = False,
    is_inference: bool = True,
    use_flash_attention: bool = False,
    use_lora: bool = False,
    lora_adapter_path: str | None = None,
    sequence_encoder_max_length: int = 2048,
    llm_max_length: int = 4096,
    structure_embedding_dim: int = 128,
    sequence_embedding_dim: int = 1280,
    llm_embedding_dim: int = 4096,
    protein_max_length: int = 2048,
):
    evo_llama_config = EvoLlamaConfig(
        structure_encoder_path=structure_encoder_path,  # type: ignore
        structure_encoder_name=structure_encoder_name,  # type: ignore
        sequence_encoder_path=sequence_encoder_path,  # type: ignore
        llm_path=llm_path,  # type: ignore
        projection_path=projection_path,  # type: ignore
        projection_fusion=projection_fusion,
        is_inference=is_inference,
        use_flash_attention=use_flash_attention,
        use_lora=use_lora,
        lora_adapter_path=lora_adapter_path,  # type: ignore
        sequence_encoder_max_length=sequence_encoder_max_length,
        llm_max_length=llm_max_length,
        structure_embedding_dim=structure_embedding_dim,
        sequence_embedding_dim=sequence_embedding_dim,
        llm_embedding_dim=llm_embedding_dim,
        protein_max_length=protein_max_length,
    )

    evo_llama = EvoLlama(evo_llama_config)
    return evo_llama


@torch.inference_mode()
def infer(
    model: EvoLlama,
    pdb_files: List[List[str]],
    sequences: List[List[str]],
    prompts: List[str],
    max_new_tokens: int = 512,
    do_sample: bool = True,
    top_p: float = 1.0,
    temperature: float = 0.2,
):
    inputs = model.prepare_inputs(pdb_files, sequences, prompts)

    outputs = model.generate(
        inputs_embeds=inputs["inputs_embeds"],
        attention_mask=inputs["attention_mask"],
        pad_token_id=model.llm.tokenizer.pad_token_id,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        use_cache=True,
        top_p=top_p,
        temperature=temperature,
    )
    responses = model.llm.tokenizer.batch_decode(
        outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )

    return responses
