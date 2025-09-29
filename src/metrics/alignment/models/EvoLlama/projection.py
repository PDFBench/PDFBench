# This file is copied from [EvoLlama](https://github.com/sornkL/EvoLlama)
# Original license: MIT License
import torch
import torch.nn as nn


class ProjectionLayer(nn.Module):
    def __init__(
        self,
        model_path: str | None = None,
        structure_embedding_dim: int = 128,
        sequence_embedding_dim: int = 1280,
        llm_embedding_dim: int = 4096,
        fusion: bool = False,
    ):
        super(ProjectionLayer, self).__init__()
        self.model_path = model_path
        self.structure_embedding_dim = structure_embedding_dim
        self.sequence_embedding_dim = sequence_embedding_dim
        self.llm_embedding_dim = llm_embedding_dim
        self.fusion = fusion
        self.structure_embedding_projection = nn.Sequential(
            nn.Linear(self.structure_embedding_dim, self.llm_embedding_dim),
            nn.ReLU(),
            nn.Linear(self.llm_embedding_dim, self.llm_embedding_dim),
        )
        self.sequence_embedding_projection = nn.Sequential(
            nn.Linear(self.sequence_embedding_dim, self.llm_embedding_dim),
            nn.ReLU(),
            nn.Linear(self.llm_embedding_dim, self.llm_embedding_dim),
        )
        self.STRUCT_EMBEDDING_PROJECTION_NAME = "structure_embedding_projection"
        self.SEQ_EMBEDDING_PROJECTION_NAME = "sequence_embedding_projection"
        self.init_projection()

    def init_projection(self):
        if self.model_path is not None:
            projection_weights = torch.load(self.model_path)  # type: ignore
            struct_embed_proj_weights = {}
            seq_embed_proj_weights = {}
            for k, v in projection_weights.items():
                key_name = k.split(".")
                key_name = ".".join(key_name[len(key_name) - 2 :])
                if self.STRUCT_EMBEDDING_PROJECTION_NAME in k:
                    struct_embed_proj_weights[key_name] = v.to(
                        next(
                            self.structure_embedding_projection.parameters()
                        ).device
                    )
                if self.SEQ_EMBEDDING_PROJECTION_NAME in k:
                    seq_embed_proj_weights[key_name] = v.to(
                        next(
                            self.sequence_embedding_projection.parameters()
                        ).device
                    )
            self.structure_embedding_projection.load_state_dict(
                struct_embed_proj_weights
            )
            self.sequence_embedding_projection.load_state_dict(
                seq_embed_proj_weights
            )
        else:
            for m in self.structure_embedding_projection.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight)
                    nn.init.zeros_(m.bias)
            for m in self.sequence_embedding_projection.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight)
                    nn.init.zeros_(m.bias)

    def forward(self, structure_representation, sequence_representation):
        structure_embedding = (
            self.structure_embedding_projection(structure_representation)
            if structure_representation is not None
            else None
        )
        sequence_embedding = (
            self.sequence_embedding_projection(sequence_representation)
            if sequence_representation is not None
            else None
        )
        if self.fusion:
            return {
                "fusion_embedding": (
                    structure_embedding + sequence_embedding  # type: ignore
                )
                if structure_embedding.shape[1] == sequence_embedding.shape[1]  # type: ignore
                else sequence_embedding
            }
        return {
            "structure_embedding": structure_embedding,
            "sequence_embedding": sequence_embedding,
        }
