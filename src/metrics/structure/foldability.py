import hashlib
import os
import warnings

import biotite.structure.io as bsio
import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import gather_object
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import EsmForProteinFolding, EsmTokenizer, logging

from src.datasets import BaseDataset
from src.metrics import BaseMetric

logging.set_verbosity_error()
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="TypedStorage is deprecated.*",
)
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message="`clean_up_tokenization_spaces` was not set.*",
)
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message="`torch.cuda.amp.autocast(args...)`",
)


def get_md5_sequence(sequence: str) -> str:
    return hashlib.md5(sequence.encode()).hexdigest()


def get_pae(output):
    pae = (
        output["aligned_confidence_probs"][0].cpu().numpy() * np.arange(64)
    ).mean(-1) * 31
    mask = output["atom37_atom_exists"][0, :, 1] == 1
    mask = mask.cpu()
    pae = pae[mask, :][:, mask]

    # PAE is a matrix with size of [L, L], representing global confidence across a sequence
    # Here we use the mean of the matrix as the global confidence score
    return pae.mean()


def compute_foldability(
    tokenizer,
    model: DistributedDataParallel,  # DDP for transformers.ESMForProteinFolding
    sequences: list[str],
    pdb_cache_dir: str,
) -> list[dict]:
    input_ids = tokenizer(
        sequences,
        return_tensors="pt",
        padding=True,
        truncation=True,
        add_special_tokens=False,
        max_length=1024,
    )["input_ids"].to(model.device)
    pae_scores = []
    with torch.no_grad():
        output = model(input_ids)
        pae_scores.append(get_pae(output))

    pdbs = model.module.output_to_pdb(output)
    md5_sequences = [get_md5_sequence(seq) for seq in sequences]

    plddt_scores = []
    for md5_sequence, pdb in zip(md5_sequences, pdbs):
        save_path = os.path.join(pdb_cache_dir, f"{md5_sequence}.pdb")
        with open(save_path, "w") as f:
            f.write(pdb)

        struct = bsio.load_structure(save_path, extra_fields=["b_factor"])
        plddt_scores.append(struct.b_factor.mean())

    ret = []
    for sequence, md5_sequence, plddt, pae in zip(
        sequences, md5_sequences, plddt_scores, pae_scores
    ):
        ret.append(
            {
                "sequence": sequence,
                "pdb_file_name": f"{md5_sequence}.pdb",
                "pLDDT": plddt,
                "pAE": pae.mean(),
            }
        )
    return ret


class FoldabilityMetric(BaseMetric):
    def __init__(self, config):
        super().__init__(config)
        self._name = config.foldability.name
        self.pdb_cache_dir = os.path.join(
            self.output_dir, config.foldability.pdb_cache_dir
        )
        self.esm_fold_name_or_path = config.foldability.esm_fold_name_or_path
        self.esm_fold_batch_size = config.foldability.esm_fold_batch_size

        os.makedirs(self.pdb_cache_dir, exist_ok=True)
        self.accelerator: Accelerator = Accelerator()

    @property
    def metrics(self) -> list[str]:
        return ["pLDDT", "pLDDT>70", "pAE", "pAE<10"]

    def _evaluate(
        self,
        dataset: BaseDataset,
    ) -> list[dict]:  # type: ignore
        # region ESM2-based BertScore
        tokenizer = EsmTokenizer.from_pretrained(self.esm_fold_name_or_path)
        model = EsmForProteinFolding.from_pretrained(self.esm_fold_name_or_path)
        model.esm = model.esm.float()
        model.trunk.set_chunk_size(64)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=1,  # TODO: support batch
            shuffle=False,
        )
        model, dataloader = self.accelerator.prepare(model, dataloader)

        all_results: list[dict] = []
        for batch in tqdm(
            dataloader,
            desc="Foldability",
            postfix=f"Batch Size: {self.esm_fold_batch_size}",
            disable=not self.accelerator.is_main_process,
        ):
            batch_size = len(batch["instruction"])
            batch_results: list[dict] = []

            # TODO: support batch
            for i in range(batch_size):
                result_item = {
                    "instruction": batch["instruction"][i],
                    "reference": batch["reference"][i],
                }
                for b in range(1, self.design_batch_size + 1):
                    result_item.update(
                        {f"response#{b}": batch[f"response#{b}"][i]}
                    )
                    _, pdb_file_name, plddt, pae = compute_foldability(
                        tokenizer,
                        model,
                        [batch[f"response#{b}"][i]],
                        self.pdb_cache_dir,
                    )[0]
                    result_item.update(
                        {
                            f"pLDDT#{b}": plddt,
                            f"pAE#{b}": pae,
                            f"pdb_file_name#{b}": pdb_file_name,
                        }
                    )
                batch_results.append(result_item)

            all_results.extend(batch_results)

        gathered_results: list[dict] = gather_object(all_results)
        # endregion

        del model, tokenizer
        torch.cuda.empty_cache()
        if self.accelerator.is_main_process:
            return gathered_results
