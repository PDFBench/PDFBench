import json
import os
from dataclasses import dataclass
from typing import Any, Optional

import simple_parsing

from .alignment_args import (
    EvoLlamaScoreArguments,
    GOScoreArguments,
    IPRScoreArguments,
    ProTrekScoreArguments,
    RetrievalAccuracyArguments,
)
from .basic_args import BasicArguments
from .launch_args import LaunchArguments
from .others_args import DiversityArguments, NoveltyArguments
from .sequence_args import (
    BertScoreArguments,
    IdentityArguments,
    PerplexityArguments,
    RepetitivenessArguments,
)
from .structure_args import FoldabilityArguments, TMScoreArguments


class Args(simple_parsing.Serializable):
    # Type hints for the arguments.
    CLS = tuple[Any, ...]

    @classmethod
    def parse(cls, args: Optional[dict[str, Any]] = None):
        assert args is None  # TODO: Load arguments from dict
        return simple_parsing.parse(
            config_class=cls,
            conflict_resolution=simple_parsing.ConflictResolution.AUTO,
            argument_generation_mode=simple_parsing.ArgumentGenerationMode.NESTED,
            add_config_path_arg=True,  # allow `--config_path`
        )

    def to_json(self):
        default = repr
        return json.dumps(self.to_dict(), indent=2, default=default)


@dataclass
class EvaluationArgs(Args):
    basic: BasicArguments
    launch: LaunchArguments
    # Sequence
    repeat: RepetitivenessArguments
    bert_score: BertScoreArguments
    identity: IdentityArguments
    perplexity: PerplexityArguments
    # Structure
    foldability: FoldabilityArguments
    tm_score: TMScoreArguments
    # Language Alignment
    protrek_score: ProTrekScoreArguments
    evollama_score: EvoLlamaScoreArguments
    ipr_score: IPRScoreArguments
    go_score: GOScoreArguments
    retrievl_acc: RetrievalAccuracyArguments
    # Others
    novelty: NoveltyArguments
    diversity: DiversityArguments

    def __post_init__(self):
        """Validation and Initialization of Arguments"""
        # mmseqs
        mmseqs_ex_path = (
            self.novelty.mmseqs_ex_path
            or self.diversity.mmseqs_ex_path
            or self.identity.mmseqs_ex_path
        )
        if mmseqs_ex_path:
            self.novelty.mmseqs_ex_path = self.diversity.mmseqs_ex_path = (
                self.identity.mmseqs_ex_path
            ) = mmseqs_ex_path
        else:
            raise ValueError(
                "At least one `mmseqs_ex_path` in "
                "[`novelty`, `diversity`, `identity`] must be set"
            )

        # tm_score
        tm_score_ex_path = (
            self.tm_score.tm_score_ex_path or self.diversity.tm_score_ex_path
        )
        if tm_score_ex_path:
            self.tm_score.tm_score_ex_path = self.diversity.tm_score_ex_path = (
                tm_score_ex_path
            )
        else:
            raise ValueError(
                "At least one `tm_score_ex_path` in "
                "[`tm_score`, `diversity`] must be set"
            )

        # pdb_cache_dir
        pdb_cache_dir = (
            self.foldability.pdb_cache_dir
            or self.tm_score.pdb_cache_dir
            or self.diversity.pdb_cache_dir
        )
        if pdb_cache_dir:
            if not os.path.isabs(pdb_cache_dir):
                pdb_cache_dir = os.path.join(
                    self.basic.output_dir, pdb_cache_dir
                )
            self.foldability.pdb_cache_dir = self.tm_score.pdb_cache_dir = (
                self.diversity.pdb_cache_dir
            ) = pdb_cache_dir
            os.makedirs(pdb_cache_dir, exist_ok=True)
        else:
            raise ValueError(
                "At least one `pdb_cache_dir` in "
                "[`foldability`, `diversity`] must be set"
            )

        # protrek
        protrek_path = (
            self.protrek_score.protrek_path or self.retrievl_acc.protrek_path
        )
        if protrek_path:
            self.protrek_score.protrek_path = self.retrievl_acc.protrek_path = (
                protrek_path
            )
        else:
            raise ValueError(
                "At least one `protrek_path` in "
                "[`protrek_score`, `retrievl_acc`] must be set"
            )
