# fmt: off
__all__ = [
    "BaseEvaluator",
    "BaseMetric",
    "EvaluationOutput",
    "MetricList",
    "BertScoreMetric", "BertScoreEvaluator",
    "RepetitivenessMetric", "RepetitivenessEvaluator",
    "PerplexityMetric", "PerplexityEvaluator",
    "IdentityMetric", "IdentityEvaluator",
    "FoldabilityMetric", "FoldabilityEvaluator",
    "TMScoreMetric", "TMScoreEvaluator",
    "ProTrekScoreMetric", "ProTrekScoreEvaluator",
    "EvoLlamaScoreMetric", "EvoLlamaScoreEvaluator",
    "RetrievalAccuracyMetric", "RetrievalAccuracyEvaluator",
    "GOScoreMetric", "GOScoreEvaluator",
    "IPRScoreMetric", "IPRScoreEvaluator",
    "DiversityMetric", "DiversityEvaluator",
    "NoveltyMetric", "NoveltyEvaluator",
]


def __getattr__(name: str):
    # base
    if name == "BaseEvaluator":
        from .metric import BaseEvaluator
        return BaseEvaluator
    if name == "BaseMetric":
        from .metric import BaseMetric
        return BaseMetric
    if name == "MetricList":
        from .metric import MetricList
        return MetricList
    if name == "EvaluationOutput":
        from .metric import EvaluationOutput
        return EvaluationOutput
    
    # sequence
    if name == "BertScoreMetric":
        from .sequence.bert_score import BertScoreMetric
        return BertScoreMetric
    if name == "BertScoreEvaluator":
        from .sequence.bert_score import BertScoreEvaluator
        return BertScoreEvaluator
    if name == "RepetitivenessMetric":
        from .sequence.repetitiveness import RepetitivenessMetric
        return RepetitivenessMetric
    if name == "RepetitivenessEvaluator":
        from .sequence.repetitiveness import RepetitivenessEvaluator
        return RepetitivenessEvaluator
    if name == "PerplexityMetric":
        from .sequence.perplexity import PerplexityMetric
        return PerplexityMetric
    if name == "PerplexityEvaluator":
        from .sequence.perplexity import PerplexityEvaluator
        return PerplexityEvaluator
    if name == "IdentityMetric":
        from .sequence.identity import IdentityMetric
        return IdentityMetric
    if name == "IdentityEvaluator":
        from .sequence.identity import IdentityEvaluator
        return IdentityEvaluator

    # structure
    if name == "FoldabilityMetric":
        from .structure.foldability import FoldabilityMetric
        return FoldabilityMetric
    if name == "FoldabilityEvaluator":
        from .structure.foldability import FoldabilityEvaluator
        return FoldabilityEvaluator
    if name == "TMScoreMetric":
        from .structure.tm_score import TMScoreMetric
        return TMScoreMetric
    if name == "TMScoreEvaluator":
        from .structure.tm_score import TMScoreEvaluator
        return TMScoreEvaluator

    # alignment
    if name == "ProTrekScoreMetric":
        from .alignment.protrek_score import ProTrekScoreMetric
        return ProTrekScoreMetric
    if name == "ProTrekScoreEvaluator":
        from .alignment.protrek_score import ProTrekScoreEvaluator
        return ProTrekScoreEvaluator
    if name == "EvoLlamaScoreMetric":
        from .alignment.evollama_score import EvoLlamaScoreMetric
        return EvoLlamaScoreMetric
    if name == "EvoLlamaScoreEvaluator":
        from .alignment.evollama_score import EvoLlamaScoreEvaluator
        return EvoLlamaScoreEvaluator
    if name == "RetrievalAccuracyMetric":
        from .alignment.retrieval_accuracy import RetrievalAccuracyMetric
        return RetrievalAccuracyMetric
    if name == "RetrievalAccuracyEvaluator":
        from .alignment.retrieval_accuracy import RetrievalAccuracyEvaluator
        return RetrievalAccuracyEvaluator
    if name == "GOScoreMetric":
        from .alignment.go_score import GOScoreMetric
        return GOScoreMetric
    if name == "GOScoreEvaluator":
        from .alignment.go_score import GOScoreEvaluator
        return GOScoreEvaluator
    if name == "IPRScoreMetric":
        from .alignment.ipr_score import IPRScoreMetric
        return IPRScoreMetric
    if name == "IPRScoreEvaluator":
        from .alignment.ipr_score import IPRScoreEvaluator
        return IPRScoreEvaluator

    # others
    if name == "DiversityMetric":
        from .others.diversity import DiversityMetric
        return DiversityMetric
    if name == "DiversityEvaluator":
        from .others.diversity import DiversityEvaluator
        return DiversityEvaluator
    if name == "NoveltyMetric":
        from .others.novelty import NoveltyMetric
        return NoveltyMetric
    if name == "NoveltyEvaluator":
        from .others.novelty import NoveltyEvaluator
        return NoveltyEvaluator

    raise AttributeError(f"module 'src.metrics' has no attribute '{name}'")
# fmt: on
