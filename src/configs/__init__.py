from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .alignment_args import (
        EvoLlamaScoreArguments,
        IPRScoreArguments,
        ProTrekScoreArguments,
        RetrievalAccuracyArguments,
    )
    from .others_args import DiversityArguments, NoveltyArguments
    from .parser import EvaluationArgs
    from .sequence_args import (
        BertScoreArguments,
        IdentityArguments,
        PerplexityArguments,
        RepetitivenessArguments,
    )
    from .structure_args import FoldabilityArguments, TMScoreArguments

__all__ = [
    "EvaluationArgs",
    "DiversityArguments",
    "NoveltyArguments",
    "BertScoreArguments",
    "IdentityArguments",
    "PerplexityArguments",
    "RepetitivenessArguments",
    "FoldabilityArguments",
    "TMScoreArguments",
    "EvoLlamaScoreArguments",
    "IPRScoreArguments",
    "ProTrekScoreArguments",
    "RetrievalAccuracyArguments",
    "RepetitivenessArguments",
    "BertScoreArguments",
    "IdentityArguments",
    "PerplexityArguments",
    "RepetitivenessArguments",
    "FoldabilityArguments",
    "TMScoreArguments",
]


def __getattr__(name: str):
    if name == "EvaluationArgs":
        from .parser import EvaluationArgs

        return EvaluationArgs
    if name == "DiversityArguments":
        from .others_args import DiversityArguments

        return DiversityArguments
    if name == "NoveltyArguments":
        from .others_args import NoveltyArguments

        return NoveltyArguments
    if name == "BertScoreArguments":
        from .sequence_args import BertScoreArguments

        return BertScoreArguments
    if name == "IdentityArguments":
        from .sequence_args import IdentityArguments

        return IdentityArguments
    if name == "PerplexityArguments":
        from .sequence_args import PerplexityArguments

        return PerplexityArguments
    if name == "RepetitivenessArguments":
        from .sequence_args import RepetitivenessArguments

        return RepetitivenessArguments
    if name == "FoldabilityArguments":
        from .structure_args import FoldabilityArguments

        return FoldabilityArguments
    if name == "TMScoreArguments":
        from .structure_args import TMScoreArguments

        return TMScoreArguments
    if name == "EvoLlamaScoreArguments":
        from .alignment_args import EvoLlamaScoreArguments

        return EvoLlamaScoreArguments
    if name == "IPRScoreArguments":
        from .alignment_args import IPRScoreArguments

        return IPRScoreArguments
    if name == "ProTrekScoreArguments":
        from .alignment_args import ProTrekScoreArguments

        return ProTrekScoreArguments
    if name == "RetrievalAccuracyArguments":
        from .alignment_args import RetrievalAccuracyArguments

        return RetrievalAccuracyArguments
    if name == "GOScoreArguments":
        from .alignment_args import GOScoreArguments

        return GOScoreArguments
    if name == "DiversityArguments":
        from .others_args import DiversityArguments

        return DiversityArguments
    if name == "NoveltyArguments":
        from .others_args import NoveltyArguments

        return NoveltyArguments
    if name == "BasicArguments":
        from .basic_args import BasicArguments

        return BasicArguments
    if name == "LaunchArguments":
        from .launch_args import LaunchArguments

        return LaunchArguments
