from enum import Enum

from .metric import BaseEvaluator, BaseMetric, EvaluationOutput, MetricList
from .sequence.bert_score import BertScoreEvaluator, BertScoreMetric
from .sequence.identity import IdentityEvaluator, IdentityMetric
from .sequence.perplexity import PerplexityEvaluator, PerplexityMetric
from .sequence.repetitiveness import (
    RepetitivenessEvaluator,
    RepetitivenessMetric,
)
from .structure.foldability import FoldabilityEvaluator, FoldabilityMetric
from .structure.tm_score import TMScoreEvaluator, TMScoreMetric
from .alignment.protrek_score import ProTrekScoreEvaluator, ProTrekScoreMetric


# class MetricType(Enum):
#     # Sequence
#     BertScoreMetric = BertScoreMetric
#     RepetitivenessMetric = RepetitivenessMetric
#     PerplexityMetric = PerplexityMetric
#     IdentityMetric = IdentityMetric

#     # Structure
#     FoldabilityMetric = FoldabilityMetric
#     # TMScoreMetric = TMScoreMetric

#     # Language Alignment
#     # ProTrekScoreMetric = ProTrekScoreMetric
#     # EvoLlamaScoreMetric = EvoLlamaScoreMetric
#     # RetrievalAccuracyMetric = RetrievalAccuracyMetric
#     # KeywordRecoveryMetric = KeywordRecoveryMetric

#     # Others
#     # DiversityMetric = DiversityMetric
#     # NoveltyMetric = NoveltyMetric
