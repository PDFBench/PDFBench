from enum import Enum
from typing import TYPE_CHECKING

__all__ = [
    "BaseDataset",
    "CostumedDataset",
    "DescriptionDataset",
    "KeywordDataset",
    "DatasetType",
]

if TYPE_CHECKING:
    from .custom import CostumedDataset
    from .dataset import BaseDataset
    from .description import DescriptionDataset
    from .keyword import KeywordDataset

    DatasetType = Enum(
        "DatasetType",
        {
            "KeywordDataset": KeywordDataset,
            "DescriptionDataset": DescriptionDataset,
            "CostumedDataset": CostumedDataset,
        },
    )


def __getattr__(name: str):
    if name == "BaseDataset":
        from .dataset import BaseDataset

        return BaseDataset

    if name == "CostumedDataset":
        from .custom import CostumedDataset

        return CostumedDataset

    if name == "DescriptionDataset":
        from .description import DescriptionDataset

        return DescriptionDataset

    if name == "KeywordDataset":
        from .keyword import KeywordDataset

        return KeywordDataset

    if name == "DatasetType":
        from .custom import CostumedDataset
        from .description import DescriptionDataset
        from .keyword import KeywordDataset

        DatasetType = Enum(
            "DatasetType",
            {
                "KeywordDataset": KeywordDataset,
                "DescriptionDataset": DescriptionDataset,
                "CostumedDataset": CostumedDataset,
            },
        )
        return DatasetType

    raise AttributeError(f"Module 'src.datasets' has no attribute '{name}'")
