import contextlib
import os
import sys

from .protrek_trimodal_model import ProTrekTrimodalModel


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
