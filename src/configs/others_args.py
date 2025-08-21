from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional


class Novelty(Enum):
    Sequence = auto()
    Structure = auto()


@dataclass
class NoveltyArguments:
    mmseqs_ex_path: Optional[str]
    novelties: Optional[tuple[Novelty, ...]] = (Novelty.Sequence,)
    run: bool = True
    name: str = "novelty"


class Diversity(Enum):
    Sequence = auto()
    Structure = auto()


@dataclass
class DiversityArguments:
    mmseqs_ex_path: Optional[str]
    tm_score_ex_path: Optional[str]
    diversities: Optional[tuple[Diversity, ...]] = (
        Diversity.Sequence,
        Diversity.Structure,
    )
    run: bool = True
    name: str = "diversity"
