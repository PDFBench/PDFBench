from dataclasses import dataclass


@dataclass
class FoldabilityArguments:
    # Whether to execute the `Foldability` evaluation.
    run: bool = True
    # File name for the output results.
    name: str = "foldability"
    # Directory for PDB cache files.
    pdb_cache_dir: str = "pdb_cache_dir/"
    # Model name or path for ESMFold. if absolute path is not provided, it will be relative to the output directory (`basic.output_dir`).
    esm_fold_name_or_path: str = "facebook/esmfold_v1"

    def __post_init__(self):
        if not self.run:
            return


@dataclass
class TMScoreArguments:
    # Whether to execute the `GT-TMScore` evaluation.
    run: bool = True
    # File name for the output results.
    name: str = "tm_score"
    # Path to the TMScore executable.
    tm_score_ex_path: str | None = None
    # Name or path for ESMFold.
    esm_fold_name_or_path: str = "facebook/esmfold_v1"
    # Directory for PDB cache files. if absolute path is not provided, it will be relative to the output directory (`basic.output_dir`).
    pdb_cache_dir: str = "pdb_cache_dir/"

    def __post_init__(self):
        if not self.run:
            return
