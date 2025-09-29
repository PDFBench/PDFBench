from dataclasses import dataclass
from enum import Enum, auto


class Repeat_Algorithm(Enum):
    RepN = auto()
    Repeat = auto()


@dataclass
class RepetitivenessArguments:
    # Whether to execute the `Repetitiveness` evaluation.
    run: bool = True
    # File name for the output results.
    name: str = "repetitiveness"
    # Algorithms used to evaluate repetitiveness.
    compute_methods: tuple[Repeat_Algorithm, ...] = (
        Repeat_Algorithm.Repeat,
        Repeat_Algorithm.RepN,
    )
    # Values of N for the RepN method.
    RepN: tuple[int, ...] = (2, 5)

    def __post_init__(self):
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
    # Whether to execute the `Perplexity` evaluation.
    run: bool = True
    # File name for the output results.
    name: str = "perplexity"
    # Models used for perplexity computation.
    compute_models: tuple[PerplexityModel, ...] = (
        PerplexityModel.ProGen2,
        PerplexityModel.ProtGPT2,
        PerplexityModel.RITA,
        PerplexityModel.ProteinGLM,
        PerplexityModel.ESMC,
    )
    # Batch size for perplexity computation.
    batch_size: int = 64
    # Model name or path for ProGen2.
    progen2_name_or_path: str = "hugohrban/progen2-base"
    # Model name or path for ProtGPT2.
    protgpt2_name_or_path: str = "nferruz/ProtGPT2"
    # Model name or path for RITA.
    rita_name_or_path: str = "lightonai/RITA_xl"
    # Model name or path for ProteinGLM. Note: perplexity values from ProteinGLM may be unstable.
    proteinglm_name_or_path: str = "biomap-research/proteinglm-3b-clm"

    def __post_init__(self):
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
    # Whether to execute the `ESMScore` evaluation.
    run: bool = True
    # File name for the output results.
    name: str = "bert_score"
    # Models used to compute ESMScore.
    compute_models: tuple[BertModel, ...] = (BertModel.ESM2,)
    # Model name or path for ESM2.
    esm2_name_or_path: str = "facebook/esm2_t33_650M_UR50D"
    # Batch size for ESM2.
    esm2_batch_size: int = 32

    def __post_init__(self):
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
    # Whether to execute the GT-Identity evaluation.
    run: bool = True
    # File name for the output results.
    name: str = "identity"
    # Number of threads used per MMseqs run.
    thread_per_mmseqs: int = 6
    # Path to the MMseqs executable.
    mmseqs_ex_path: str | None = None

    def __post_init__(self):
        if not self.run:
            return
