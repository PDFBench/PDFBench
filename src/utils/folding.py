import hashlib
import os
import warnings

import biotite.structure.io as bsio
import numpy as np
import torch
from transformers import EsmForProteinFolding, EsmTokenizer, logging

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


def seq_to_md5(sequence: str) -> str:
    return hashlib.md5(sequence.encode()).hexdigest()


def output_to_pae(output):
    pae = (
        output["aligned_confidence_probs"][0].cpu().numpy() * np.arange(64)
    ).mean(-1) * 31
    mask = output["atom37_atom_exists"][0, :, 1] == 1
    mask = mask.cpu()
    pae = pae[mask, :][:, mask]

    # PAE is a matrix with size of [L, L], representing global confidence across a sequence
    # Here we use the mean of the matrix as the global confidence score
    return pae.mean()


def clean_seq(sequence: str) -> str:
    replacement_map = {
        "B": "N",  # Aspartic acid or Asparagine -> Asparagine
        "Z": "Q",  # Glutamic acid or Glutamine -> Glutamine
        "J": "L",  # Leucine or Isoleucine -> Leucine
        "U": "C",  # Selenocysteine -> Cysteine
        "O": "K",  # Pyrrolysine -> Lysine
        "X": "G",  # Unknown -> Glycine
        "-": "",  # Gap -> Remove
        "*": "",  # Stop codon -> Remove
        "?": "G",  # Unknown -> Glycine
        "~": "G",  # Unresolved -> Glycine
        ".": "",  # Gap -> Remove
    }

    # 将序列转换为大写，并处理小写字母（先转换再替换）
    cleaned_seq = sequence.upper()

    # 进行替换
    for non_std, replacement in replacement_map.items():
        cleaned_seq = cleaned_seq.replace(non_std, replacement)

    return cleaned_seq


def seq_to_struc(
    tokenizer: EsmTokenizer,
    model: EsmForProteinFolding,
    sequences: list[str],
    pdb_cache_dir: str,
    return_foldability: bool = True,
) -> list[dict]:
    if return_foldability:
        return compute_foldability(
            tokenizer,
            model,
            sequences,
            pdb_cache_dir,
        )

    md5_sequences = [seq_to_md5(seq) for seq in sequences]

    existing_seqs = [
        md5_seq
        for md5_seq in md5_sequences
        if os.path.exists(os.path.join(pdb_cache_dir, f"{md5_seq}.pdb"))
    ]
    existing_seqs = set(existing_seqs)

    sequences_to_compute = []
    indices_to_compute = []
    for idx, (seq, md5_seq) in enumerate(zip(sequences, md5_sequences)):
        if md5_seq not in existing_seqs:
            sequences_to_compute.append(seq)
            indices_to_compute.append(idx)

    sequences_to_compute = [clean_seq(seq) for seq in sequences_to_compute]
    if sequences_to_compute:
        input_ids = tokenizer(
            sequences_to_compute,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=False,
            max_length=1024,
        )["input_ids"].to(model.device)  # type: ignore

        with torch.no_grad():
            output = model(input_ids)

        pdbs = model.output_to_pdb(output)
        for idx, pdb in zip(indices_to_compute, pdbs):
            md5_seq = md5_sequences[idx]
            save_path = os.path.join(pdb_cache_dir, f"{md5_seq}.pdb")
            with open(save_path, "w") as f:
                f.write(pdb)

    ret = []
    for sequence, md5_sequence in zip(sequences, md5_sequences):
        ret.append(
            {
                "sequence": sequence,
                "pdb_file_name": f"{md5_sequence}.pdb",
            }
        )
    return ret


def compute_foldability(
    tokenizer,
    model: EsmForProteinFolding,
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
        pae_scores.append(output_to_pae(output))

    pdbs = model.output_to_pdb(output)
    md5_sequences = [seq_to_md5(seq) for seq in sequences]

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
