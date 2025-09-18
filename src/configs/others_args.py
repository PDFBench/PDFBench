import os
from dataclasses import dataclass
from enum import Enum, auto


class Novelty(Enum):
    Sequence = auto()
    Structure = auto()


@dataclass
class NoveltyArguments:
    mmseqs_ex_path: str
    foldseek_ex_path: str
    mmseqs_targetdb_path: str
    foldseek_targetdb_path: str
    run: bool = True
    name: str = "novelty"
    workers_per_mmseqs: int | None = -1
    workers_per_foldseek: int | None = -1
    novelties: tuple[Novelty, ...] = (Novelty.Sequence,)
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
    mmseqs_ex_path: str | None = None
    tm_score_ex_path: str | None = None
    esm_fold_name_or_path: str = "facebook/esmfold_v1"
    pdb_cache_dir: str = "pdb_cache_dir/"
    diversities: tuple[Diversity, ...] = (
        Diversity.Sequence,
        Diversity.Structure,
    )
    run: bool = True
    name: str = "diversity"

    def __post_init__(self):
        if not self.run:
            return

        if not self.diversities:
            raise ValueError(
                "At least one diversity (`Sequence`, `Structure`) must be selected for computing Diversity"
            )
