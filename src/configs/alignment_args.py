from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

import psutil


@dataclass
class ProTrekScoreArguments:
    protrek_path: str
    run: bool = True
    name: str = "protrek_score"


@dataclass
class EvoLlamaScoreArguments:
    evollama_path: str
    llama_name_or_path: Optional[str] = "meta-llama/Llama-3.2-3B-Instruct"
    pubmedbert_name_or_path: Optional[str] = "NeuML/pubmedbert-base-embeddings"
    run: bool = True
    name: str = "evollama_score"


@dataclass
class KeywordRecoveryArguments:
    interpro_scan_ex: str
    workers_per_interpro_scan: Optional[int] = psutil.cpu_count(logical=True)
    run: bool = True
    name: str = "keyword_recovery"


class RetrievalDifficulty(Enum):
    Easy = auto()
    Medium = auto()
    Hard = auto()


@dataclass
class RetrievalAccuracyArguments:
    retrieval_difficulties: Optional[tuple[RetrievalDifficulty, ...]] = (
        RetrievalDifficulty.Easy,
        RetrievalDifficulty.Medium,
        RetrievalDifficulty.Hard,
    )
    molinst_pool: Optional[str] = (
        "/home/jhkuang/data/cache/dynamsa/data/Molinst/inst2seq.json"
    )
    esmw_pool: Optional[str] = (
        "/home/jhkuang/data/cache/dynamsa/data/UniInPro/Inst2seq_w.small.json"
    )
    esmwo_pool: Optional[str] = (
        "/home/jhkuang/data/cache/dynamsa/data/UniInPro/Inst2seq_wo.small.json"
    )
    ec_pool: Optional[str] = (
        "/home/jhkuang/data/cache/dynamsa/data/SwissEC/SwissEC_Pool.json"
    )
    run: bool = True
    name: str = "retrieval_accuracy"
