from dataclasses import dataclass


@dataclass
class FoldabilityArguments:
    run: bool = True
    name: str = "foldability"
    pdb_cache_dir: str = "pdb_cache_dir"
    esm_fold_name_or_path: str = "EvolutionaryScale/esm3-sm-open-v1"

    def init(self):
        if not self.run:
            return


@dataclass
class TMScoreArguments:
    run: bool = True
    name: str = "tm_score"
    tm_score_ex_path: str | None = None

    def init(self):
        if not self.run:
            return
