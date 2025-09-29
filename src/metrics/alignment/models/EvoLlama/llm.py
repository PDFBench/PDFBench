# This file is copied from [EvoLlama](https://github.com/sornkL/EvoLlama)
# Original license: MIT License
from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from peft import LoraConfig, PeftModelForCausalLM
from transformers import (
    Cache,
    GenerationConfig,
    LlamaForCausalLM,
    LlamaTokenizerFast,
    LogitsProcessorList,
    PreTrainedModel,
    StoppingCriteriaList,
)
from transformers.generation.streamers import BaseStreamer
from transformers.generation.utils import GenerateOutput
from transformers.modeling_outputs import CausalLMOutputWithPast

from .constants import *


class LLM(nn.Module):
    def __init__(
        self,
        model_path: str,
        model_max_length: int = 4096,
        is_inference: bool = True,
        use_flash_attention: bool = False,
        use_lora: bool = False,
        lora_adapter_path: str = None,
    ):
        super(LLM, self).__init__()
        self.model_path = model_path
        self.model_max_length = model_max_length
        self.is_inference = is_inference
        self.use_flash_attention = use_flash_attention
        self.use_lora = use_lora
        self.lora_adapter_path = lora_adapter_path
        self.tokenizer = self.init_tokenizer()
        if self.is_inference:
            self.model = LlamaForCausalLM.from_pretrained(
                self.model_path,
                device_map="cpu",
                use_flash_attention_2=self.use_flash_attention,
                torch_dtype="auto",
            )
            self.tokenizer.padding_side = "left"
        else:
            self.model = LlamaForCausalLM.from_pretrained(
                self.model_path,
                use_flash_attention_2=self.use_flash_attention,
                torch_dtype=torch.bfloat16,
            )
            self.tokenizer.padding_side = "right"
        self.model.resize_token_embeddings(len(self.tokenizer))
        if self.use_lora:
            lora_config = LoraConfig.from_pretrained(self.lora_adapter_path)
            self.model = PeftModelForCausalLM.from_pretrained(
                self.model,
                self.lora_adapter_path,
                is_trainable=not self.is_inference,
                config=lora_config,
            )

    def init_tokenizer(self):
        tokenizer = LlamaTokenizerFast.from_pretrained(
            self.model_path, model_max_length=self.model_max_length
        )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        additional_special_tokens = {
            "additional_special_tokens": [
                STRUCTURE_BEGIN_TOKEN,
                STRUCTURE_END_TOKEN,
                SEQUENCE_BEGIN_TOKEN,
                SEQUENCE_END_TOKEN,
                SEPERATOR,
            ]
        }
        tokenizer.add_special_tokens(
            additional_special_tokens, replace_additional_special_tokens=False
        )
        return tokenizer

    def construct_conversations(
        self, user_prompts: List[str], assistant_responses: List[str] = None
    ) -> List[str]:
        add_generation_prompt = True
        if assistant_responses is not None:
            assert len(user_prompts) == len(assistant_responses)
            add_generation_prompt = False
        else:
            assistant_responses = [None for _ in range(len(user_prompts))]

        conversations = []
        for prompt, response in zip(user_prompts, assistant_responses):
            conversation = [
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": prompt},
            ]
            if response is not None:
                conversation.append({"role": "assistant", "content": response})
            conversations.append(
                self.tokenizer.apply_chat_template(
                    conversation,
                    tokenize=False,
                    add_generation_prompt=add_generation_prompt,
                )
            )

        return conversations

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def get_output_embeddings(self):
        return self.model.get_output_embeddings()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        prefix_allowed_tokens_fn: Optional[
            Callable[[int, torch.Tensor], List[int]]
        ] = None,
        synced_gpus: Optional[bool] = None,
        assistant_model: Optional[PreTrainedModel] = None,
        streamer: Optional[BaseStreamer] = None,
        negative_prompt_ids: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        return self.model.generate(
            inputs=inputs,
            generation_config=generation_config,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            synced_gpus=synced_gpus,
            assistant_model=assistant_model,
            streamer=streamer,
            negative_prompt_ids=negative_prompt_ids,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            **kwargs,
        )
