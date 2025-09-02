from enum import Enum

from .metric import BaseEvaluator, BaseMetric, EvaluationOutput, MetricList
from .sequence.bert_score import BertScoreEvaluator, BertScoreMetric
from .sequence.identity import IdentityMetric
from .sequence.perplexity import PerplexityMetric
from .sequence.repetitiveness import RepetitivenessMetric
from .structure.foldability import FoldabilityMetric


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
