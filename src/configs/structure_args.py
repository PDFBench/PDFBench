from dataclasses import dataclass


@dataclass
class StructureArguments:
    esm_fold: str
    pdb_cache_dir: str
    tm_score_ex: str
    workers_per_tm_score: int
