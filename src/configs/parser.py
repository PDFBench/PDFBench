import json
from dataclasses import dataclass, fields
from typing import Any, Optional

import simple_parsing

from .alignment_args import (
    EvoLlamaScoreArguments,
    KeywordRecoveryArguments,
    ProTrekScoreArguments,
    RetrievalAccuracyArguments,
)
from .basic_args import BasicArguments
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
        # simple_parsing.helpers.serialization.from_dict()
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
    keyword_rec: KeywordRecoveryArguments
    retrievl_acc: RetrievalAccuracyArguments
    # Others
    novelty: NoveltyArguments
    diversity: DiversityArguments

    def init(
        self,
    ):
        """Validation and Initialization of Arguments"""
        for field in fields(self):
            sub_args = getattr(self, field.name)
            sub_args.init()

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
