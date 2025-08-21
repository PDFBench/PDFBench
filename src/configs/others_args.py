from dataclasses import dataclass
from enum import Enum, auto


class Novelty(Enum):
    Sequence = auto()
    Structure = auto()


@dataclass
class NoveltyArguments:
    mmseqs_ex_path: str | None = None
    novelties: tuple[Novelty, ...] = (Novelty.Sequence,)
    run: bool = True
    name: str = "novelty"

    def init(self):
        if not self.run:
            return

        if not self.novelties:
            raise ValueError(
                "At least one novelty (`Sequence`, `Structure`) must be selected for computing Novelty"
            )


class Diversity(Enum):
    Sequence = auto()
    Structure = auto()


@dataclass
class DiversityArguments:
    mmseqs_ex_path: str | None = None
    tm_score_ex_path: str | None = None
    diversities: tuple[Diversity, ...] = (
        Diversity.Sequence,
        Diversity.Structure,
    )
    run: bool = True
    name: str = "diversity"

    def init(self):
        if not self.run:
            return

        if not self.diversities:
            raise ValueError(
                "At least one diversity (`Sequence`, `Structure`) must be selected for computing Diversity"
            )
