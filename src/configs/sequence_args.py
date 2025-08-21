from dataclasses import dataclass
from enum import Enum, auto


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

    def init(self):
        if not self.run:
            return

        if not self.compute_methods:
            raise ValueError(
                "At least one method (`Repeat`, `RepN`) must be selected for computing Repetitiveness"
            )
        if Repeat_Algorithm.RepN in self.compute_methods:
            if not self.RepN:
                raise ValueError(
                    "At least one RepN must be selected for computing Repetitiveness"
                )
            if min(self.RepN) < 2 or max(self.RepN) > 20:
                raise ValueError(
                    "RepN must be in the range [2, 20] for computing Repetitiveness"
                )


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
    compute_models: tuple[PerplexityModel, ...] = (
        PerplexityModel.ProGen2,
        PerplexityModel.ProtGPT2,
        PerplexityModel.RITA,
        PerplexityModel.ProteinGLM,
        PerplexityModel.ESMC,
    )
    progen2_name_or_path: str = "hugohrban/progen2-base"
    protgpt2_name_or_path: str = "nferruz/ProtGPT2"
    rita_name_or_path: str = "lightonai/RITA_xl"
    proteinglm_name_or_path: str = "biomap-research/proteinglm-3b-clm"

    def init(self):
        if not self.run:
            return

        if not self.compute_models:
            raise ValueError(
                "At least one model "
                "(`ProGen2`, `ProtGPT2`, `RITA`, `ProteinGLM`, `ESMC`) "
                "must be selected for computing Perplexity"
            )


class BertModel(Enum):
    ESM2 = auto()


@dataclass
class BertScoreArguments:
    run: bool = True
    name: str = "bert_score"
    compute_models: tuple[BertModel, ...] = (BertModel.ESM2,)
    esm2_name_or_path: str = "facebook/esm2_t33_650M_UR50D"

    def init(self):
        if not self.run:
            return

        if not self.compute_models:
            raise ValueError(
                "At least one model "
                "(`ESM2`) "
                "must be selected for computing Perplexity"
            )


@dataclass
class IdentityArguments:
    run: bool = True
    name: str = "identity"
    mmseqs_ex_path: str | None = None

    def init(self):
        if not self.run:
            return
