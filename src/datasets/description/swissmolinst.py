import re

from ..dataset import BaseDataset


class SwissMolinstDataset(BaseDataset):
    """
    Description-guided Dataset
    for
    [SwissMolinst](?)
    and [MolinstTest](?) with description
    from
    [SwissProtCLAP](https://huggingface.co/datasets/vinesmsuic/SwissProtCLAP)
    and protein_design.json of [Mol-Instructions](https://huggingface.co/datasets/zjunlp/Mol-Instructions),
    used in PDFBench official experiments.
    """

    # TODO: add link

    @classmethod
    def function(cls, instruction: str) -> str:
        """
        Extract the part of Instruciton related to protein function

        :param instruciton: Instruction of InterProDataset
        :type instruciton: str
        :return: Part of the instruction containing protein function in InterProDataset
        :rtype: str
        """
        function = re.sub(r"^.*?(1\.)", r"\1", instruction)
        function = function.removesuffix("The designed protein sequence is ")
        return function.strip()
