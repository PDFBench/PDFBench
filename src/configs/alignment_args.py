import os
import shutil
from dataclasses import dataclass
from enum import Enum, auto

from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ProTrekScoreArguments:
    protrek_path: str | None = None
    run: bool = True
    name: str = "protrek_score"

    def __post_init__(self):
        if not self.run:
            return


@dataclass
class EvoLlamaScoreArguments:
    evollama_path: str
    llama_name_or_path: str = "meta-llama/Llama-3.2-3B-Instruct"
    pubmedbert_name_or_path: str = "NeuML/pubmedbert-base-embeddings"
    run: bool = True
    name: str = "evollama_score"

    def __post_init__(self):
        if not self.run:
            return


class Keyword(Enum):
    IPR = auto()
    GO = auto()
    EC = auto()


@dataclass
class IPRScoreArguments:
    # InterPro
    interpro_scan_ex_path: str
    interpro_cache_path: str
    workers_per_scan: int

    run: bool = True
    name: str = "ipr_score"

    def __post_init__(self):
        if not self.run:
            return

        if self.workers_per_scan == -1:
            cpu_cnt: int | None = os.cpu_count()
            assert cpu_cnt is not None, (
                "Python.os cannot detect cpu count of your device, "
                "please set num_cpu manually"
            )
            self.workers_per_scan = cpu_cnt if cpu_cnt < 16 else cpu_cnt - 8

        if shutil.which("java") is None:
            self.run = False
            logger.warning_rank0("Java is not installed, skip IPRScoreMetric")


@dataclass
class GOScoreArguments:
    deepgo_root: str
    deepgo_weight_path: str
    deepgo_threshold: float = 0.7
    deepgo_batch_size: int = 64

    run: bool = True
    name: str = "go_score"

    def __post_init__(self):
        if not self.run:
            return


class RetrievalDifficulty(Enum):
    Soft = auto()
    Normal = auto()
    Hard = auto()


@dataclass
class RetrievalAccuracyArguments:
    protrek_path: str | None = None
    protrek_batch_size: int | None = 16
    retrieval_difficulties: tuple[RetrievalDifficulty, ...] = (
        RetrievalDifficulty.Soft,
        RetrievalDifficulty.Normal,
        RetrievalDifficulty.Hard,
    )
    run: bool = True
    name: str = "retrieval_accuracy"

    def __post_init__(self):
        if not self.run:
            return

        if not self.retrieval_difficulties:
            raise ValueError(
                "At least one difficulty (`Easy`, `Medium`, `Hard`) must be selected for computing Retrieval Accuracy"
            )
