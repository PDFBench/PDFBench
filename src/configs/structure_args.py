from dataclasses import dataclass
from typing import Optional


@dataclass
class FoldabilityArguments:
    run: bool = True
    name: str = "foldability"
    pdb_cache_dir: str = "pdb_cache_dir"
    esm_fold_name_or_path: Optional[str] = "EvolutionaryScale/esm3-sm-open-v1"


@dataclass
class TMScoreArguments:
    run: bool = True
    name: str = "tm_score"
    tm_score_ex_path: Optional[str] = None
