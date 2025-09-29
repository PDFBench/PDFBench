import os
import shutil
from dataclasses import dataclass
from enum import Enum, auto

from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ProTrekScoreArguments:
    # Path to ProTrek-650M weights.
    protrek_path: str | None = None
    # Whether to execute the `ProTrek Score` evaluation.
    run: bool = True
    # File name for the output results.
    name: str = "protrek_score"

    def __post_init__(self):
        if not self.run:
            return


@dataclass
class EvoLlamaScoreArguments:
    # Path to `EvoLlama Score` weights.
    evollama_path: str
    # Model name or path for `Llama-3.2-3B-Instruct`.
    llama_name_or_path: str = "meta-llama/Llama-3.2-3B-Instruct"
    # Model name or path for `PubMedBERT embeddings`.
    pubmedbert_name_or_path: str = "NeuML/pubmedbert-base-embeddings"
    # Whether to execute the `EvoLlama Score` evaluation.
    run: bool = True
    # File name for the output results.
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
    # Path to the InterProScan executable.
    interpro_scan_ex_path: str
    # Path to the InterProScan cache directory. if absolute path is not provided, it will be relative to the output directory (`basic.output_dir`).
    interpro_cache_path: str
    # Number of workers used for InterProScan. if -1, PDFBench will use all CPU cores - 8 if CPU cores > 32, otherwise use all CPU cores. We recommend setting this value to -1.
    workers_per_scan: int = -1

    # Whether to execute the `IPR Recovery` evaluation.
    run: bool = True
    # File name for the output results.
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
    # Path to DeepGO-SE weights.
    deepgo_weight_path: str
    # Confidence threshold for GO prediction.
    deepgo_threshold: float = 0.7
    # Batch size for DeepGO-SE.
    deepgo_batch_size: int = 64

    # Whether to execute the `GO Recovery` evaluation.
    run: bool = True
    # File name for the output results.
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
    # Path to ProTrek Score weights.
    protrek_path: str | None = None
    # Batch size for ProTrek evaluation.
    protrek_batch_size: int | None = 16
    # Difficulty levels for retrieval accuracy evaluation.
    retrieval_difficulties: tuple[RetrievalDifficulty, ...] = (
        RetrievalDifficulty.Soft,
        RetrievalDifficulty.Normal,
        RetrievalDifficulty.Hard,
    )
    # Whether to execute the `Retrieval Accuracy` evaluation.
    run: bool = True
    # File name for the output results.
    name: str = "retrieval_accuracy"

    def __post_init__(self):
        if not self.run:
            return

        if not self.retrieval_difficulties:
            raise ValueError(
                "At least one difficulty (`Easy`, `Medium`, `Hard`) must be selected for computing Retrieval Accuracy"
            )
