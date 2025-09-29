import os

from ...utils.context_manager import suppress_all_output

with suppress_all_output():
    from .models.ProTrek.protrek_trimodal_model import (
        ProTrekTrimodalModel,
    )


def load_protrek(protrek_path: str, device_id: int) -> ProTrekTrimodalModel:
    config = {
        "protein_config": os.path.join(protrek_path, "esm2_t33_650M_UR50D"),
        "text_config": os.path.join(
            protrek_path,
            "BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        ),
        "structure_config": os.path.join(protrek_path, "foldseek_t30_150M"),
        "load_protein_pretrained": False,
        "load_text_pretrained": False,
        "from_checkpoint": os.path.join(
            protrek_path, "ProTrek_650M_UniRef50.pt"
        ),
    }
    with suppress_all_output():
        return ProTrekTrimodalModel(**config).eval().to(f"cuda:{device_id}")  # type: ignore
