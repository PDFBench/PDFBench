from enum import Enum

from src.datasets import ECNumberDataset, InterProDataset, SwissMolinstDataset
from src.metrics import (
    BertScoreMetric,
    FoldabilityMetric,
    IdentityMetric,
    PerplexityMetric,
    RepetitivenessMetric,
)
