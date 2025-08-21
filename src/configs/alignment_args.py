from dataclasses import dataclass
from enum import Enum, auto


@dataclass
class ProTrekScoreArguments:
    protrek_path: str
    run: bool = True
    name: str = "protrek_score"

    def init(self):
        if not self.run:
            return


@dataclass
class EvoLlamaScoreArguments:
    evollama_path: str
    llama_name_or_path: str = "meta-llama/Llama-3.2-3B-Instruct"
    pubmedbert_name_or_path: str = "NeuML/pubmedbert-base-embeddings"
    run: bool = True
    name: str = "evollama_score"

    def init(self):
        if not self.run:
            return


@dataclass
class KeywordRecoveryArguments:
    interpro_scan_ex: str
    workers_per_interpro_scan: int | None = -1
    run: bool = True
    name: str = "keyword_recovery"

    def init(self):
        if not self.run:
            return


class RetrievalDifficulty(Enum):
    Easy = auto()
    Medium = auto()
    Hard = auto()


@dataclass
class RetrievalAccuracyArguments:
    retrieval_difficulties: tuple[RetrievalDifficulty, ...] = (
        RetrievalDifficulty.Easy,
        RetrievalDifficulty.Medium,
        RetrievalDifficulty.Hard,
    )
    molinst_pool: str = (
        "/home/jhkuang/data/cache/dynamsa/data/Molinst/inst2seq.json"
    )
    interpro_pool: str = (
        "/home/jhkuang/data/cache/dynamsa/data/UniInPro/Inst2seq_w.small.json"
    )
    ec_pool: str = (
        "/home/jhkuang/data/cache/dynamsa/data/SwissEC/SwissEC_Pool.json"
    )
    run: bool = True
    name: str = "retrieval_accuracy"

    def init(self):
        if not self.run:
            return

        if not self.retrieval_difficulties:
            raise ValueError(
                "At least one difficulty (`Easy`, `Medium`, `Hard`) must be selected for computing Retrieval Accuracy"
            )
