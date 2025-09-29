# This file is copied from [EvoLlama](https://github.com/sornkL/EvoLlama)
# Original license: MIT License
import copy
from typing import List, Union

import torch
import torch.nn as nn

# from rdkit import Chem
# from torchdrug import models, layers, data
# from torchdrug.layers import geometry
from .protein_mpnn_utils import (
    ProteinMPNN,
    StructureDatasetPDB,
    parse_PDB,
    tied_featurize,
)

# This is used to prevent the error: Attribute Error: xxx is not a torch.nn.Module
# Ref: https://github.com/DeepGraphLearning/torchdrug/issues/77
# nn.Module = nn._Module


class ProteinMPNNStructureEncoder(nn.Module):
    def __init__(self, model_path: str):
        super(ProteinMPNNStructureEncoder, self).__init__()
        self.model_path = model_path
        self.hidden_dim = 128
        self.num_layers = 3
        self.num_letters = 21
        self.num_edges = 48
        self.ca_only = False
        self.model_checkpoint = torch.load(self.model_path, map_location="cpu")
        if "model_state_dict" in self.model_checkpoint.keys():
            self.model_checkpoint = self.model_checkpoint["model_state_dict"]
        self.model = ProteinMPNN(
            num_letters=self.num_letters,
            node_features=self.hidden_dim,
            edge_features=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            num_encoder_layers=self.num_layers,
            num_decoder_layers=self.num_layers,
            k_neighbors=self.num_edges,
            augment_eps=0.0,  # set augment_eps to 0.0 to disable randomness
        )
        new_model_checkpoint = {}
        for key in self.model.state_dict().keys():
            for checkpoint_key in self.model_checkpoint.keys():
                if key in checkpoint_key:
                    new_model_checkpoint[key] = self.model_checkpoint[
                        checkpoint_key
                    ]
        self.model.load_state_dict(new_model_checkpoint)

    def forward(self, pdb_files: List[Union[str, StructureDatasetPDB]]):
        model_device = next(self.model.parameters()).device
        structure_representations = []
        for pdb_file in pdb_files:
            if isinstance(pdb_file, StructureDatasetPDB):
                dataset = pdb_file
            elif isinstance(pdb_file, str):
                pdb_dict_list = parse_PDB(pdb_file, ca_only=self.ca_only)
                dataset = StructureDatasetPDB(pdb_dict_list, max_length=200000)
            else:
                raise ValueError(
                    "pdb_files should be a list of str (lazy-preprocess) or StructureDatasetPDB (preprocessed)"
                )

            all_chain_list = [
                item[-1:]
                for item in list(pdb_dict_list[0])
                if item[:9] == "seq_chain"
            ]  # ['A','B', 'C',...]
            designed_chain_list = all_chain_list
            fixed_chain_list = [
                letter
                for letter in all_chain_list
                if letter not in designed_chain_list
            ]
            chain_id_dict = {}
            chain_id_dict[pdb_dict_list[0]["name"]] = (
                designed_chain_list,
                fixed_chain_list,
            )

            protein = dataset[0]
            batch_clones = [copy.deepcopy(protein)]
            (
                X,
                _,
                mask,
                lengths,
                _,
                chain_encoding_all,
                _,
                _,
                _,
                _,
                _,
                _,
                residue_idx,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
            ) = tied_featurize(
                batch_clones,
                model_device,
                chain_id_dict,
                None,
                None,
                None,
                None,
                None,
                ca_only=self.ca_only,
            )

            structure_representation = self.model(
                X, _, mask, _, residue_idx, chain_encoding_all, _
            )["node_representation"]
            attention_mask = torch.ones(
                structure_representation.shape[1], device=model_device
            )

            structure_representations.append(
                {
                    "representation": structure_representation,
                    "attention_mask": attention_mask,
                }
            )

        max_node_length = max(
            [
                representation["representation"].shape[1]
                for representation in structure_representations
            ]
        )
        for representation in structure_representations:
            representation["representation"] = torch.cat(
                [
                    representation["representation"],
                    torch.zeros(
                        (
                            representation["representation"].shape[0],
                            max_node_length
                            - representation["representation"].shape[1],
                            representation["representation"].shape[2],
                        ),
                        device=model_device,
                    ),
                ],
                dim=1,
            )
            representation["representation"] = representation[
                "representation"
            ].view(-1, self.hidden_dim)
            representation["attention_mask"] = torch.cat(
                [
                    representation["attention_mask"],
                    torch.zeros(
                        max_node_length
                        - representation["attention_mask"].shape[0],
                        device=model_device,
                    ),
                ],
                dim=0,
            )
        structure_representations = {
            k: torch.stack(
                [
                    representation[k]
                    for representation in structure_representations
                ]
            )
            .detach()
            .cpu()
            for k in structure_representations[0].keys()
        }

        return structure_representations


class GearNetStructureEncoder(nn.Module):
    pass
