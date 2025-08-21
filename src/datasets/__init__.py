from enum import Enum
from .dataset import BaseDataset
from .description.swissmolinst import SwissMolinstDataset
from .keyword.ecnumber import ECNumberDataset
from .keyword.interpro import InterProDataset


class DatasetType(Enum):
    InterPro = InterProDataset
    ECNumber = ECNumberDataset
    SwissMolinst = SwissMolinstDataset
