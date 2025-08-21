import json
from dataclasses import dataclass
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

    def validate(
        self,
    ):
        pass
