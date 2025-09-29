import re

from .dataset import BaseDataset


class DescriptionDataset(BaseDataset):
    """
    Description-guided Dataset
    for
    [SwissMolinst](https://huggingface.co/datasets/nwliu/Molinst-SwissProtCLAP)
    and [MolinstTest](https://huggingface.co/datasets/nwliu/Molinst-SwissProtCLAP) with description
    from
    [SwissProtCLAP](https://huggingface.co/datasets/vinesmsuic/SwissProtCLAP)
    and protein_design.json of [Mol-Instructions](https://huggingface.co/datasets/zjunlp/Mol-Instructions),
    used in PDFBench official experiments.
    """

    @classmethod
    def function(cls, instruction: str) -> str:
        """
        Extract the part of Instruction related to protein function

        :param instruction: Instruction of InterProDataset
        :type instruction: str
        :return: Part of the instruction containing protein function in InterProDataset
        :rtype: str
        """
        function = re.sub(r"^.*?(1\.)", r"\1", instruction)
        function = function.removesuffix("The designed protein sequence is ")
        return function.strip()
