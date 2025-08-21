from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional


class Repeat_Algorithm(Enum):
    RepN = auto()
    Repeat = auto()


@dataclass
class RepetitivenessArguments:
    run: bool = True
    name: str = "repetitiveness"
    compute_methods: tuple[Repeat_Algorithm, ...] = (
        Repeat_Algorithm.Repeat,
        Repeat_Algorithm.RepN,
    )
    RepN: tuple[int, ...] = (2, 5)


class PerplexityModel(Enum):
    ProGen2 = auto()
    ProtGPT2 = auto()
    RITA = auto()
    ProteinGLM = auto()
    ESMC = auto()


@dataclass
class PerplexityArguments:
    run: bool = True
    name: str = "perplexity"
    compute_models: Optional[tuple[PerplexityModel, ...]] = (
        PerplexityModel.ProGen2,
        PerplexityModel.ProtGPT2,
        PerplexityModel.RITA,
        PerplexityModel.ProteinGLM,
        PerplexityModel.ESMC,
    )
    progen2_name_or_path: Optional[str] = "hugohrban/progen2-base"
    protgpt2_name_or_path: Optional[str] = "nferruz/ProtGPT2"
    rita_name_or_path: Optional[str] = "lightonai/RITA_xl"
    proteinglm_name_or_path: Optional[str] = "biomap-research/proteinglm-3b-clm"


class BertModel(Enum):
    ESM2 = auto()


@dataclass
class BertScoreArguments:
    run: bool = True
    name: str = "bert_score"
    compute_models: Optional[tuple[BertModel, ...]] = (BertModel.ESM2,)
    esm2_name_or_path: Optional[str] = "facebook/esm2_t33_650M_UR50D"


@dataclass
class IdentityArguments:
    run: bool = True
    name: str = "identity"
    mmseqs_ex_path: Optional[str] = None
