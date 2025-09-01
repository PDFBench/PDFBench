from dataclasses import dataclass


@dataclass
class FoldabilityArguments:
    run: bool = True
    name: str = "foldability"
    pdb_cache_dir: str = "pdb_cache_dir"
    esm_fold_name_or_path: str = "facebook/esmfold_v1"
    esm_fold_batch_size: int = 1  # TODO: support batch

    def __post_init__(self):
        if not self.run:
            return


@dataclass
class TMScoreArguments:
    run: bool = True
    name: str = "tm_score"
    tm_score_ex_path: str | None = None

    def __post_init__(self):
        if not self.run:
            return
