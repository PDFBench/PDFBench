import os
from dataclasses import dataclass
from enum import Enum, auto


class Novelty(Enum):
    Sequence = auto()
    Structure = auto()


@dataclass
class NoveltyArguments:
    # Path to the MMseqs executable. Please refer to `README.md` for the installation of MMseqs.
    mmseqs_ex_path: str
    # Path to the Foldseek executable. Please refer to `README.md` for the installation of Foldseek.
    foldseek_ex_path: str
    # Path to the MMseqs search database. Please refer to `README.md` for the building of MMseqs search database.
    mmseqs_targetdb_path: str
    # Path to the Foldseek search database. Please refer to `README.md` for the building of Foldseek search database.
    foldseek_targetdb_path: str
    # Whether to execute the `Novelty` evaluation.
    run: bool = True
    # File name for the output results.
    name: str = "novelty"
    # Number of workers used for MMseqs. if -1, PDFBench will use all CPU cores - 8 if CPU cores > 32, otherwise use all CPU cores. We recommend setting this value to -1.
    workers_per_mmseqs: int | None = -1
    # Number of workers used for Foldseek. if -1, PDFBench will use all CPU cores - 8 if CPU cores > 32, otherwise use all CPU cores. We recommend setting this value to -1.
    workers_per_foldseek: int | None = -1
    # Types of novelty considered.
    novelties: tuple[Novelty, ...] = (Novelty.Sequence,)
    # Directory for PDB cache files. if absolute path is not provided, it will be relative to the output directory (`basic.output_dir`).
    pdb_cache_dir: str = "pdb_cache_dir/"

    def __post_init__(self):
        if not self.run:
            return

        if not self.novelties:
            raise ValueError(
                "At least one novelty (`Sequence`, `Structure`) must be selected for computing Novelty"
            )

        if self.workers_per_mmseqs == -1:
            cpu_cnt = os.cpu_count()
            assert cpu_cnt is not None
            self.workers_per_mmseqs = cpu_cnt - 8 if cpu_cnt > 32 else cpu_cnt

        if self.workers_per_foldseek == -1:
            cpu_cnt = os.cpu_count()
            assert cpu_cnt is not None
            self.workers_per_foldseek = cpu_cnt - 8 if cpu_cnt > 32 else cpu_cnt


class Diversity(Enum):
    Sequence = auto()
    Structure = auto()


@dataclass
class DiversityArguments:
    # Path to the MMseqs executable.
    mmseqs_ex_path: str | None = None
    # Path to the TMScore executable.
    tm_score_ex_path: str | None = None
    # Model name or path for ESMFold.
    esm_fold_name_or_path: str = "facebook/esmfold_v1"
    # Directory for PDB cache files. if absolute path is not provided, it will be relative to the output directory (`basic.output_dir`).
    pdb_cache_dir: str = "pdb_cache_dir/"
    # Types of diversity considered.
    diversities: tuple[Diversity, ...] = (
        Diversity.Sequence,
        Diversity.Structure,
    )
    # Whether to execute the `Diversity` evaluation.
    run: bool = True
    # File name for the output results.
    name: str = "diversity"

    def __post_init__(self):
        if not self.run:
            return

        if not self.diversities:
            raise ValueError(
                "At least one diversity (`Sequence`, `Structure`) must be selected for computing Diversity"
            )
