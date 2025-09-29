# This file is copied from [EvoLlama](https://github.com/sornkL/EvoLlama)
# Original license: MIT License
import contextlib
import os
import sys
import warnings
from typing import Callable, List, Literal, Optional, Tuple, Union

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
)
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from transformers import (
    Cache,
    GenerationConfig,
    LogitsProcessorList,
    PretrainedConfig,
    PreTrainedModel,
    StoppingCriteriaList,
)
from transformers.generation.streamers import BaseStreamer
from transformers.generation.utils import GenerateOutput
from transformers.modeling_outputs import CausalLMOutputWithPast

from .constants import *
from .llm import LLM
from .projection import ProjectionLayer
from .sequence_encoder import SequenceEncoder
from .structure_encoder import (
    GearNetStructureEncoder,
    ProteinMPNNStructureEncoder,
)


@contextlib.contextmanager
def suppress_all_output():
    """上下文管理器，用于同时抑制标准输出和标准错误输出"""
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


class EvoLlamaConfig(PretrainedConfig):
    model_type = "evollama"

    def __init__(
        self,
        structure_encoder_path: str = None,
        structure_encoder_name: Literal[
            "ProteinMPNN", "GearNet"
        ] = "ProteinMPNN",
        sequence_encoder_path: str = None,
        llm_path: str = None,
        projection_path: str = None,
        projection_fusion: bool = False,
        is_inference: bool = True,
        use_flash_attention: bool = False,
        use_lora: bool = False,
        lora_adapter_path: str = None,
        sequence_encoder_max_length: int = 2048,
        llm_max_length: int = 4096,
        structure_embedding_dim: int = 128,
        sequence_embedding_dim: int = 1280,
        llm_embedding_dim: int = 4096,
        protein_max_length: int = 2048,
        **kwargs,
    ):
        super(EvoLlamaConfig, self).__init__(**kwargs)
        self.structure_encoder_path = structure_encoder_path
        self.structure_encoder_name = structure_encoder_name
        self.sequence_encoder_path = sequence_encoder_path
        self.llm_path = llm_path
        self.projection_path = projection_path
        self.projection_fusion = projection_fusion
        self.is_inference = is_inference
        self.use_flash_attention = use_flash_attention
        self.use_lora = use_lora
        self.lora_adapter_path = lora_adapter_path
        self.sequence_encoder_max_length = sequence_encoder_max_length
        self.llm_max_length = llm_max_length
        self.structure_embedding_dim = structure_embedding_dim
        self.sequence_embedding_dim = sequence_embedding_dim
        self.llm_embedding_dim = llm_embedding_dim
        self.protein_max_length = protein_max_length


class EvoLlama(PreTrainedModel):
    config_class = EvoLlamaConfig
    supports_gradient_checkpointing = True

    def __init__(self, config: EvoLlamaConfig):
        super(EvoLlama, self).__init__(config)
        with suppress_all_output():
            self.config = config
            if config.structure_encoder_name == STRUCTURE_ENCODER_PROTEIN_MPNN:
                self.structure_encoder = ProteinMPNNStructureEncoder(
                    config.structure_encoder_path
                )
            elif config.structure_encoder_name == STRUCTURE_ENCODER_GEARNET:
                self.structure_encoder = GearNetStructureEncoder(
                    config.structure_encoder_path
                )
            else:
                raise NotImplementedError(
                    f"{config.structure_encoder_name} is not supported."
                )
            self.sequence_encoder = SequenceEncoder(
                config.sequence_encoder_path, config.sequence_encoder_max_length
            )
            self.llm = LLM(
                config.llm_path,
                config.llm_max_length,
                config.is_inference,
                config.use_flash_attention,
                config.use_lora,
                config.lora_adapter_path,
            )
            self.projection = ProjectionLayer(
                config.projection_path,
                config.structure_embedding_dim,
                config.sequence_embedding_dim,
                config.llm_embedding_dim,
                config.projection_fusion,
            )
            if self.config.is_inference:
                self.structure_encoder = self.structure_encoder.to(
                    self.llm.model.device
                )
                self.sequence_encoder = self.sequence_encoder.to(
                    self.llm.model.device
                )
                self.projection = self.projection.to(self.llm.model.device)
            self.system_message_length = self.get_system_message_length()

    def get_input_embeddings(self) -> nn.Module:
        return self.llm.get_input_embeddings()

    def get_output_embeddings(self) -> nn.Module:
        return self.llm.get_output_embeddings()

    def get_system_message_length(self):
        conversations = self.llm.tokenizer.apply_chat_template(
            conversation=[
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": ""},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )
        conversation_inputs = self.llm.tokenizer(conversations)
        system_message_length = len(conversation_inputs["input_ids"]) - 1
        return system_message_length

    def insert_protein_embeddings(
        self,
        structures: List[Optional[torch.Tensor]],
        sequences: List[Optional[torch.Tensor]],
        conversation: torch.Tensor,
        padding_offset: int = 0,
    ):
        # If the padding side is left, then skipping the padding before inserting the embedding
        protein_token_offset = 5  # <end> <sep> <start>: </structure> <sep> <sequence> or </sequence> <sep> <structure> or </protein> <sep> <protein>
        offset = self.system_message_length + padding_offset
        for struct, seq in zip(structures, sequences):
            if struct is not None:
                conversation = torch.cat(
                    [
                        conversation[:, :offset],
                        struct,
                        conversation[:, offset:],
                    ],
                    dim=1,
                )
                offset += struct.shape[1] + protein_token_offset
            if seq is not None:
                conversation = torch.cat(
                    [conversation[:, :offset], seq, conversation[:, offset:]],
                    dim=1,
                )
                offset += seq.shape[1] + protein_token_offset
        return conversation

    def insert_position_ids(
        self,
        structures: List[Optional[torch.Tensor]],
        sequences: List[Optional[torch.Tensor]],
        conversation_length: int,
    ):
        position_ids = torch.arange(self.system_message_length).unsqueeze(0)
        for idx, (struct, seq) in enumerate(zip(structures, sequences)):
            if idx != 0:  # add position_ids of <sep>
                sep_position_ids = torch.full(
                    (position_ids.shape[0], 3), position_ids[0, -1].item() + 1
                )
                position_ids = torch.cat(
                    [position_ids, sep_position_ids], dim=1
                )
            if struct is not None:
                struct_position_ids = torch.full(
                    (struct.shape[0], struct.shape[1] + 2),
                    position_ids[0, -1].item() + 1,
                )
                position_ids = torch.cat(
                    [position_ids, struct_position_ids], dim=1
                )
            if seq is not None:
                sep_position_ids = torch.full(
                    (seq.shape[0], 3), position_ids[0, -1].item() + 1
                )
                position_ids = torch.cat(
                    [position_ids, sep_position_ids], dim=1
                )
                seq_position_ids = torch.full(
                    (seq.shape[0], seq.shape[1] + 2),
                    position_ids[0, -1].item() + 1,
                )
                position_ids = torch.cat(
                    [position_ids, seq_position_ids], dim=1
                )

        padding_start_position_id = position_ids[0, -1].item() + 1
        padding_position_ids = torch.arange(
            padding_start_position_id,
            conversation_length
            - position_ids.shape[1]
            + padding_start_position_id,
        )
        position_ids = torch.cat(
            [position_ids, padding_position_ids.unsqueeze(0)], dim=1
        )
        return position_ids.to(self.llm.model.device)

    def pad_tensors(
        self, batch: List[torch.Tensor], pad_value: int = 0
    ) -> torch.Tensor:
        max_length = min(
            self.config.llm_max_length,
            max([tensor.shape[1] for tensor in batch]),
        )
        padded_batch = []
        for tensor in batch:
            if tensor.shape[1] < max_length:
                padding_shape = (
                    tensor.shape[0],
                    max_length - tensor.shape[1],
                    *tensor.shape[2:],
                )
                padding_tensor = torch.full(
                    padding_shape,
                    pad_value,
                    device=tensor.device,
                    dtype=tensor.dtype,
                )
                if self.llm.tokenizer.padding_side == "right":
                    padded_tensor = torch.cat([tensor, padding_tensor], dim=1)
                else:
                    padded_tensor = torch.cat([padding_tensor, tensor], dim=1)
            else:
                padded_tensor = tensor
            padded_batch.append(padded_tensor)

        return torch.cat(padded_batch, dim=0)

    def prepare_inputs(
        self,
        pdb_files: Optional[List[List[str]]],
        sequences: Optional[List[List[str]]],
        prompts: List[str],
        responses: List[str] = None,
    ):
        return_labels = True
        if pdb_files is not None:
            assert len(pdb_files) == len(prompts), (
                f"{len(pdb_files)} != {len(prompts)}"
            )
        if sequences is not None:
            assert len(sequences) == len(prompts), (
                f"{len(sequences)} != {len(prompts)}"
            )
        if (pdb_files is None and self.config.projection_fusion) or (
            sequences is None and self.config.projection_fusion
        ):
            raise ValueError(
                "Projection fusion is only supported when both pdb_files and sequences are provided."
            )
        if pdb_files is None and sequences is None:
            raise ValueError("Either pdb_files or sequences must be provided.")
        if responses is not None:
            assert len(prompts) == len(responses)
        else:
            return_labels = False
            responses = [None for _ in range(len(prompts))]

        # Stack pdb_files and sequences respectively, and get embeddings of them
        protein_count = (
            [len(pdb_file) for pdb_file in pdb_files]
            if pdb_files is not None
            else [len(sequence) for sequence in sequences]
        )
        struct_representations, struct_embeddings, struct_attention_masks = (
            None,
            None,
            None,
        )
        seq_representations, seq_embeddings, seq_attention_masks = (
            None,
            None,
            None,
        )

        if pdb_files is not None:
            pdb_files = [
                pdb_file for pdb_files in pdb_files for pdb_file in pdb_files
            ]
            struct_outputs = self.structure_encoder(pdb_files)
            struct_representations = struct_outputs["representation"].to(
                next(self.projection.parameters()).device
            )
            struct_attention_masks = struct_outputs["attention_mask"].to(
                self.llm.model.device
            )
        if sequences is not None:
            sequences = [
                sequence for sequences in sequences for sequence in sequences
            ]
            seq_outputs = self.sequence_encoder(sequences)
            seq_representations = seq_outputs["representation"].to(
                next(self.projection.parameters()).device
            )
            seq_attention_masks = seq_outputs["attention_mask"].to(
                self.llm.model.device
            )

        embeddings = self.projection(
            struct_representations, seq_representations
        )
        if self.config.projection_fusion:
            fusion_embeddings = embeddings["fusion_embedding"].to(
                self.llm.model.device
            )

            # truncate if the embeddings and attention masks exceed the protein_max_length
            fusion_embeddings = fusion_embeddings[
                :, : self.config.protein_max_length
            ]
            fusion_attention_masks = seq_attention_masks[
                :, : self.config.protein_max_length
            ]
        else:
            if pdb_files is not None:
                struct_embeddings = embeddings["structure_embedding"].to(
                    self.llm.model.device
                )
                # truncate if the embeddings and attention masks exceed the protein_max_length
                struct_embeddings = struct_embeddings[
                    :, : self.config.protein_max_length
                ]
                struct_attention_masks = struct_attention_masks[
                    :, : self.config.protein_max_length
                ]
            if sequences is not None:
                seq_embeddings = embeddings["sequence_embedding"].to(
                    self.llm.model.device
                )
                # truncate if the embeddings and attention masks exceed the protein_max_length
                seq_embeddings = seq_embeddings[
                    :, : self.config.protein_max_length
                ]
                seq_attention_masks = seq_attention_masks[
                    :, : self.config.protein_max_length
                ]

            # trick: regard the struct/seq as fusion if struct is None or seq is None
            if pdb_files is None or sequences is None:
                fusion_embeddings = (
                    struct_embeddings
                    if seq_embeddings is None
                    else seq_embeddings
                )
                fusion_attention_masks = (
                    struct_attention_masks
                    if seq_attention_masks is None
                    else seq_attention_masks
                )

        # Get embeddings/attention_masks/position_ids/labels of conversations, including proteins inserted
        inserted_conv_embeddings = []
        inserted_conv_attention_masks = []
        inserted_conv_position_ids = []
        inserted_conv_labels = []
        begin_idx = 0
        for protein_cnt, prompt, response in zip(
            protein_count, prompts, responses
        ):
            if (
                self.config.projection_fusion
                or pdb_files is None
                or sequences is None
            ):
                prompt_prefix = f"{PROTEIN_BEGIN_TOKEN}{PROTEIN_END_TOKEN}"
            else:
                prompt_prefix = f"{STRUCTURE_BEGIN_TOKEN}{STRUCTURE_END_TOKEN} {SEPERATOR} {SEQUENCE_BEGIN_TOKEN}{SEQUENCE_END_TOKEN}"
            new_prompt = f" {SEPERATOR} ".join(
                [prompt_prefix for _ in range(protein_cnt)]
            )
            new_prompt = f"{new_prompt} {prompt}"
            if response is None:
                conv = self.llm.construct_conversations([new_prompt])
            else:
                conv = self.llm.construct_conversations(
                    [new_prompt], [response]
                )
            conv_inputs = self.llm.tokenizer(
                conv, return_tensors="pt", padding=True, truncation=True
            ).to(self.llm.model.device)
            conv_input_ids = conv_inputs["input_ids"]
            conv_embeddings = self.get_input_embeddings()(conv_input_ids).to(
                self.llm.model.device
            )
            conv_attention_masks = conv_inputs["attention_mask"]

            end_idx = begin_idx + protein_cnt
            if (
                self.config.projection_fusion
                or pdb_files is None
                or sequences is None
            ):
                target_protein_embeddings = fusion_embeddings[begin_idx:end_idx]
                inserted_conv_embeddings.append(
                    self.insert_protein_embeddings(
                        [
                            embedding.unsqueeze(0)
                            for embedding in target_protein_embeddings
                        ],
                        [None for _ in range(protein_cnt)],
                        conv_embeddings,
                    )
                )
                inserted_conv_attention_masks.append(
                    self.insert_protein_embeddings(
                        [
                            mask.unsqueeze(0)
                            for mask in fusion_attention_masks[
                                begin_idx:end_idx
                            ]
                        ],
                        [None for _ in range(protein_cnt)],
                        conv_attention_masks,
                    )
                )
                inserted_conv_position_ids.append(
                    self.insert_position_ids(
                        [
                            mask.unsqueeze(0)
                            for mask in fusion_attention_masks[
                                begin_idx:end_idx
                            ]
                        ],
                        [None for _ in range(protein_cnt)],
                        inserted_conv_attention_masks[-1].shape[1],
                    )
                )
            else:
                target_struct_embeddings = struct_embeddings[begin_idx:end_idx]
                target_seq_embeddings = seq_embeddings[begin_idx:end_idx]
                inserted_conv_embeddings.append(
                    self.insert_protein_embeddings(
                        [
                            embedding.unsqueeze(0)
                            for embedding in target_struct_embeddings
                        ],
                        [
                            embedding.unsqueeze(0)
                            for embedding in target_seq_embeddings
                        ],
                        conv_embeddings,
                    )
                )
                inserted_conv_attention_masks.append(
                    self.insert_protein_embeddings(
                        [
                            mask.unsqueeze(0)
                            for mask in struct_attention_masks[
                                begin_idx:end_idx
                            ]
                        ],
                        [
                            mask.unsqueeze(0)
                            for mask in seq_attention_masks[begin_idx:end_idx]
                        ],
                        conv_attention_masks,
                    )
                )
                inserted_conv_position_ids.append(
                    self.insert_position_ids(
                        [
                            embedding.unsqueeze(0)
                            for embedding in struct_attention_masks[
                                begin_idx:end_idx
                            ]
                        ],
                        [
                            embedding.unsqueeze(0)
                            for embedding in seq_attention_masks[
                                begin_idx:end_idx
                            ]
                        ],
                        inserted_conv_attention_masks[-1].shape[1],
                    )
                )
            if response is not None:
                conv_wo_response = self.llm.construct_conversations(
                    [new_prompt]
                )
                conv_wo_response_inputs = self.llm.tokenizer(
                    conv_wo_response,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                ).to(self.llm.model.device)
                conv_wo_response_input_ids = conv_wo_response_inputs[
                    "input_ids"
                ]
                conv_labels = torch.empty_like(conv_input_ids)
                conv_labels[:, : conv_wo_response_input_ids.shape[1]] = (
                    IGNORE_INDEX
                )
                conv_labels[:, conv_wo_response_input_ids.shape[1] :] = (
                    conv_input_ids[:, conv_wo_response_input_ids.shape[1] :]
                )
                conv_labels_rows = conv_labels.shape[0]
                if (
                    self.config.projection_fusion
                    or pdb_files is None
                    or sequences is None
                ):
                    conv_labels_cols = (
                        target_protein_embeddings.shape[0]
                        * target_protein_embeddings.shape[1]
                    )
                else:
                    conv_labels_cols = (
                        target_struct_embeddings.shape[0]
                        * target_struct_embeddings.shape[1]
                        + target_seq_embeddings.shape[0]
                        * target_seq_embeddings.shape[1]
                    )
                inserted_conv_labels.append(
                    torch.cat(
                        [
                            torch.full(
                                (conv_labels_rows, conv_labels_cols),
                                IGNORE_INDEX,
                                device=conv_labels.device,
                                dtype=conv_labels.dtype,
                            ),
                            conv_labels,
                        ],
                        dim=1,
                    )
                )
            begin_idx = end_idx

        # Padding the embeddings, attention masks and position_ids
        inserted_conv_embeddings = self.pad_tensors(inserted_conv_embeddings)
        inserted_conv_attention_masks = self.pad_tensors(
            inserted_conv_attention_masks
        )
        inserted_conv_position_ids = self.pad_tensors(
            inserted_conv_position_ids, pad_value=IGNORE_POSITION_ID
        )

        if not return_labels:
            return {
                "inputs_embeds": inserted_conv_embeddings,
                "attention_mask": inserted_conv_attention_masks,
                "position_ids": inserted_conv_position_ids,
            }
        inserted_conv_labels = self.pad_tensors(
            inserted_conv_labels, pad_value=IGNORE_INDEX
        )
        return {
            "inputs_embeds": inserted_conv_embeddings,
            "attention_mask": inserted_conv_attention_masks,
            "position_ids": inserted_conv_position_ids,
            "labels": inserted_conv_labels,
        }

    @autocast()
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
        return self.llm(
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

    @autocast()
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
        assistant_model: Optional["PreTrainedModel"] = None,
        streamer: Optional[BaseStreamer] = None,
        negative_prompt_ids: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        return self.llm.generate(
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


if __name__ == "__main__":

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    from src.config import config

    evo_llama_config = EvoLlamaConfig(
        structure_encoder_path=config.structure_encoder_path,
        sequence_encoder_path=config.sequence_encoder_path,
        llm_path=config.llm_path,
        is_inference=True,
        protein_max_length=2,
        projection_fusion=False,
    )
    evo_llama = EvoLlama(evo_llama_config)
    evo_llama.eval()

    pdb_files = [
        [f"{config.pdb_dir_path}/77acda85e7fd580aedab745b9cae6e8a.pdb"],
        # [f'{config.pdb_dir_path}/601df2792f8b4c18e987eb27dfe98d14.pdb'],
        # [f'{config.pdb_dir_path}/83bcb0271760f2d342e60083b99e3000.pdb', f'{config.pdb_dir_path}/feddc59753f809aaec08b7135fe3f889.pdb']
    ]
    # pdb_files = None
    sequences = [
        [
            "GMILKTNLFGHTYQFKSITDVLAKANEEKSGDRLAGVAAESAEERVAAKVVLSKMTLGDLRNNPVVPYETDEVTRIIQDQVNDRIHDSIKNWTVEELREWILDHKTTDADIKRVARGLTSEIIAAVTKLMSNLDLIYGAKKIRVIAHANTTIGLPGTFSARLQPNHPTDDPDGILASLMEGLTYGIGDAVIGLNPVDDSTDSVVRLLNKFEEFRSKWDVPTQTCVLAHVKTQMEAMRRGAPTGLVFQSIAGSEKGNTAFGFDGATIEEARQLALQSGAATGPNVMYFETGQGSELSSDAHFGVDQVTMEARCYGFAKKFDPFLVNTVVGFIGPEYLYDSKQVIRAGLEDHFMGKLTGISMGCDVCYTNHMKADQNDVENLSVLLTAAGCNFIMGIPHGDDVMLNYQTTGYHETATLRELFGLKPIKEFDQWMEKMGFSENGKLTSRAGDASIFLK"
        ],
        # ['MAHHHHHHMSFFRMKRRLNFVVKRGIEELWENSFLDNNVDMKKIEYSKTGDAWPCVLLRKKSFEDLHKLYYICLKEKNKLLGEQYFHLQNSTKMLQHGRLKKVKLTMKRILTVLSRRAIHDQCLRAKDMLKKQEEREFYEIQKFKLNEQLLCLKHKMNILKKYNSFSLEQISLTFSIKKIENKIQQIDIILNPLRKETMYLLIPHFKYQRKYSDLPGFISWKKQNIIALRNNMSKLHRLY'],
        # ['MSYEKKSVEGAYIDDSTTFEAFHLDSRLLQAIKNIGFQYPTLIQSHAIPLALQQKRDIIAKAATGSGKTLAYLIPVIETILEYKKTIDNGEENGTLGIILVPTRELAQQVYNVLEKLVLYCSKDIRTLNISSDMSDSVLSTLLMDQPEIIVGTPGKLLDLLQTKINSISLNELKFLVVDEVDLVLTFGYQDDLNKIGEYLPLKKNLQTFLMSATLNDDIQALKQKFCRSPAILKFNDEEINKNQNKLLQYYVKVSEFDKFLLCYVIFKLNLIKGKTLIFVNNIDRGYRLKLVMEQFGIKSCILNSELPVNSRQHIVDQFNKNVYQLLIATDDTEYIKEEDDEIEEGHNTENQEEKSLEGEPENDKKPSKKKKVQVKKDKEYGVSRGVDFKNVACVLNFDLPTTAKSYVHRVGRTARGGKTGTAISFVVPLKEFGKHKPSMLQTAKKDERILSRIIKQQSKLGLELQPYKFDQKQVEAFRYRMEDGFRAVTQVAIREARVKELKQELLASEKLKRHFEENPKELQSLRHDKELHPARVQQHLKRVPDYLLPESARGNGTKVKFVPFHNAKKRHSHKKGRVSKPKNGKVDPLKNFK', 'MADEERLKEFKEANKIVFDPNTRQVWENQNRDGTKPATTFQSEEDIKRAAPESEKDTSATSGIVPTLQNIVATVTLGCRLDLKTVALHARNAEYNPKRFAAVIMRIREPKTTALIFASGKMVVTGAKSEDDSKLASRKYARIIQKIGFAAKFTDFKIQNIVGSCDVKFPIRLEGLAFSHGTFSSYEPELFPGLIYRMVKPKIVLLIFVSGKIVLTGAKQREEIYQAFEAIYPVLSEFRKM']
    ]
    # sequences = None
    prompts = [
        ""
        # 'Identify if the given protein possesses solubility properties.\nAnswer the question using a single word or phrase.',
        # 'Would you classify the given protein as soluble?\nAnswer the question using a single word or phrase.',
        # 'Predict if the two yeast protein sequences interact with each other. Answer the question using a single word or phrase.'
    ]
    responses = [
        "Soluble.\n",
        # 'Not soluble.\n',
        # 'Yes.\n'
    ]
    inputs = evo_llama.prepare_inputs(pdb_files, sequences, prompts, responses)
    ic(inputs, {k: v.shape for k, v in inputs.items()})
    outputs = evo_llama(**inputs)
