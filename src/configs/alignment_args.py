import os
from dataclasses import dataclass
from enum import Enum, auto


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


@dataclass
class KeywordRecoveryArguments:
    interpro_scan_ex_path: str
    workers_per_scan: int = -1
    run: bool = True
    name: str = "keyword_recovery"

    def __post_init__(self):
        if not self.run:
            return

        if self.workers_per_scan == -1:
            cpu_count: int | None = os.cpu_count()
            assert cpu_count is not None, (
                "Python.os cannot detect cpu count of your device, "
                "please set num_cpu manually"
            )
            self.workers_per_scan = cpu_count


class RetrievalDifficulty(Enum):
    Easy = auto()
    Medium = auto()
    Hard = auto()


@dataclass
class RetrievalAccuracyArguments:
    protrek_path: str | None = None
    protrek_batch_size: int | None = 64
    retrieval_difficulties: tuple[RetrievalDifficulty, ...] = (
        RetrievalDifficulty.Easy,
        RetrievalDifficulty.Medium,
        RetrievalDifficulty.Hard,
    )
    desc_pool: str = (
        "/home/jhkuang/data/cache/dynamsa/data/Molinst/inst2seq.json"
    )
    ipr_pool: str = "/nas/data/jhkuang/data/cache/dynamsa/data/keyword_guided/SwissIPG/out/swissipg_ipr.json"
    go_pool: str = "/nas/data/jhkuang/data/cache/dynamsa/data/keyword_guided/SwissIPG/out/swissipg_go.json"
    ipr_go_pool: str = "/nas/data/jhkuang/data/cache/dynamsa/data/keyword_guided/SwissIPG/out/swissipg_ipr_go.json"
    ec_pool: str = (
        "/home/jhkuang/data/cache/dynamsa/data/SwissEC/SwissEC_Pool.json"
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
