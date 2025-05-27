import json
import multiprocessing as mp
import os
import subprocess
import tempfile
import warnings
from typing import List

import numpy as np
from tqdm.auto import tqdm


def process_m8_file(file_path, n_prot=3):
    similaritys = []
    with open(file_path, "r") as file:
        for line in file:
            parts = line.strip().split("\t")
            assert len(parts) > 3, "MMSeqs2 M8 should have at least 3 columns"
            query_id, match_id = parts[0], parts[1]
            if query_id == match_id:
                continue

            similarity = float(parts[2])
            similaritys.append(similarity)

    total = n_prot * (n_prot - 1)
    hits = sum(similaritys)
    dismiss = (total - len(similaritys)) * 1
    diversity = (hits + dismiss) / total

    return diversity


def mmseqs_easy_search(
    mmseqs_path: str,
    sequences: list[str],
    fasta_file: str,
    result_m8_file: str,
    temp_folder: str,
):
    # --- 1. Prepare FASTA file ---
    with open(fasta_file, "w") as f:
        for i, seq in enumerate(sequences):
            f.write(f">seq_{i}\n")
            fasta_sequence = "\n".join(
                [seq[j : j + 60] for j in range(0, len(seq), 60)]
            )
            f.write(f"{fasta_sequence}\n")

    # --- 2. Run MMseqs2 ---
    args = [
        mmseqs_path,
        "easy-search",
        fasta_file,
        fasta_file,
        result_m8_file,
        temp_folder,
        "-v",
        "1",
        "--remove-tmp-files",
        "1",
        "--threads",
        "6",
        "-e",
        "1000000",
    ]
    return subprocess.run(args)


def compute_diversity(
    sequences: List[str],
    mmseqs_path: str,
) -> float:
    """
    Computes diversity within a list of sequences using MMseqs2.

    Diversity is defined as the mean dissimilarity (1 - similarity)
    between all unique pairs of sequences in the input list.

    :param List[str] sequences: A list of protein sequences.
    :param str temp_folder_base: Base directory for temporary files.
                                 A unique subfolder will be created inside this.
    :param str mmseqs_path: Path to the MMseqs2 executable.
    :return float: The computed diversity value (average dissimilarity).
                   Returns 0.0 if less than 2 sequences are provided.
    """
    assert len(sequences) >= 2, "Diversity requires at least two sequences."

    with tempfile.TemporaryDirectory() as temp_folder:
        fasta_file = os.path.join(temp_folder, "sequences.fasta")
        result_m8_file = os.path.join(temp_folder, "result.m8")

        res = mmseqs_easy_search(
            mmseqs_path,
            sequences,
            fasta_file,
            result_m8_file,
            temp_folder,
        )
        if res.returncode != 0:
            warnings.warn("mmseqs easy-search failed with {sequences}")
            return 1.0
        diversity = process_m8_file(result_m8_file, n_prot=len(sequences))

    return diversity


def _main(
    uid: int,
    queue: mp.Queue,
    subset: list,
    num_epoch: int,
    mmseqs_path: str,
):
    results: list = [dict() for _ in range(len(subset))]

    idx = 0
    for item in tqdm(
        subset,
        desc=f"Process {uid} - Diversity",
        position=uid + 1,
        ncols=100,
    ):
        sequences = [
            item[f"response_{epoch}"] for epoch in range(1, num_epoch + 1)
        ]

        res: dict = {
            "diversity": compute_diversity(
                sequences,
                mmseqs_path,
            ),
        }
        results[idx].update(res)
        idx += 1

    queue.put(results)


def main(
    num_epoches: int,
    eval_dir: str,
    sequence_file_name: str,
    evaluation_file: str,
    mmseqs_path: str,
    num_workers: int,
):
    assert num_epoches and eval_dir and evaluation_file and mmseqs_path

    if not os.path.exists(evaluation_file):
        mp.set_start_method("spawn", force=True)

        # collect data
        data = []
        for epoch in range(1, num_epoches + 1):
            sequence_file = os.path.join(
                eval_dir, str(epoch), sequence_file_name
            )
            assert os.path.exists(sequence_file), (
                f"{sequence_file} does not exist"
            )
            with open(sequence_file, "r") as f:
                _data = json.load(f)

            if data == []:  # epoch:1
                data = [
                    {f"response_{epoch}": value["response"]} for value in _data
                ]
            else:
                for idx in range(len(data)):
                    data[idx].update(
                        {f"response_{epoch}": _data[idx]["response"]}
                    )

        queue: mp.Queue = mp.Queue()
        processes: list = []
        for i in range(num_workers):
            piece = len(data) // num_workers
            beg_idx = i * piece
            end_idx = (i + 1) * piece if i != num_workers - 1 else len(data)
            subset = data[beg_idx:end_idx]

            p = mp.Process(
                target=_main,
                args=(
                    i,
                    queue,
                    subset,
                    num_epoches,
                    mmseqs_path,
                ),
            )
            p.start()
            processes.append(p)

        results: list = [queue.get() for _ in range(len(processes))]
        results = [element for sublist in results for element in sublist]

        for p in processes:
            p.join()

        with open(evaluation_file, "w") as f:
            json.dump(results, f, indent=4)  # type: ignore

    else:
        print("Load processed evaluation file")
        with open(evaluation_file, "r") as f:
            results: list = json.load(f)

    support_metrics = [
        "diversity",
    ]
    for metric in support_metrics:
        mean = np.mean([sample[metric] for sample in results])
        print(f"mean {metric}: {mean:.2f}")


def test_example_01():
    print("Test Example 01 -> Three natural sequences from UniProtKB")
    seq01 = "MLLLLLLLLLLPPLVLRVAASRCLHDETQKSVSLLRPPFSQLPSKSRSSSLTLPSSRDPQPLRIQSCYLGDHISDGAWDPEGEGMRGGSRALAAVREATQRIQAVLAVQGPLLLSRDPAQYCHAVWGDPDSPNYHRCSLLNPGYKGESCLGAKIPDTHLRGYALWPEQGPPQLVQPDGPGVQNTDFLLYVRVAHTSKCHQETVSLCCPGWSTAAQSQLTAALTSWAQRRGFVMLPRLCLKLLGSSNLPTLASQSIRITGPSVIAYAACCQLDSEDRPLAGTIVYCAQHLTSPSLSHSDIVMATLHELLHALGFSGQLFKKWRDCPSGFSVRENCSTRQLVTRQDEWGQLLLTTPAVSLSLAKHLGVSGASLGVPLEEEEGLLSSHWEARLLQGSLMTATFDGAQRTRLDPITLAAFKDSGWYQVNHSAAEELLWGQGSGPEFGLVTTCGTGSSDFFCTGSGLGCHYLHLDKGSCSSDPMLEGCRMYKPLANGSECWKKENGFPAGVDNPHGEIYHPQSRCFFANLTSQLLPGDKPRHPSLTPHLKEAELMGRCYLHQCTGRGAYKVQVEGSPWVPCLPGKVIQIPGYYGLLFCPRGRLCQTNEDINAVTSPPVSLSTPDPLFQLSLELAGPPGHSLGKEQQEGLAEAVLEALASKGGTGRCYFHGPSITTSLVFTVHMWKSPGCQGPSVATLHKALTLTLQKKPLEVYHGGANFTTQPSKLLVTSDHNPSMTHLRLSMGLCLMLLILVGVMGTTAYQKRATLPVRPSASYHSPELHSTRVPVRGIREV"
    seq02 = "MAGIIKKQILKHLSRFTKNLSPDKINLSTLKGEGELKNLELDEEVLQNMLDLPTWLAINKVFCNKASIRIPWTKLKTHPICLSLDKVIMEMSTCEEPRSPNGPSPIATASGQSEYGFAEKVVEGISVSVNSIVIRIGAKAFNASFELSQLRIYSVNAHWEHGDLRFTRIQDPQRGEVLTFKEINWQMIRIEADATQSSHLEIMCAPVRLITNQSKIRVTLKRRLKDCNVIATKLVLILDDLLWVLTDSQLKAMVQYAKSLSEAIEKSTEQRKSMAPEPTQSSTVVASAQQVKTTQTSNAPDVNDAIVKLFNDFDVKETSHHLVISHLDLHICDDIHAKEKESNRRITGGAMQLSFTQLTIDYYPYHKAGDSCNHWMYFSDATKTKNGWANELLHEFECNVEMLKQAVKDHNVGSPPKSPTHASPQHTQTEKDYPLKGTCRTPSVLSQQSKAKLMSSSVVVRLADFNIYQVSTAEQCRSSPKSMICCNKKSLYLPQEMSAVYIEFTEYYYPDGKDFPIPSPNLYSQLNALQFTVDERSILWLNQFLLDLKQSLNQFMAVYKLNDNSKSDEHVDVRVDGLMLKFVIPSEVKSECHQDQPRAISIQSSEMIATNTRHCPNCRHSDLEALFQDFKDCDFFSKTYTSFPKSCDNFNLLHPIFQRHAHEQDTKMHEIYKGNITPQLNKNTLKTSAATDVWAVYFSQFWIDYEGMKSGKGRPISFVDSFPLSIWICQPTRYAESQKEPQTCNQVSLNTSQSESSDLAGRLKRKKLLKEYYSTESEPLTNGGQKPSSSDTFFRFSPSSSEADIHLLVHVHKHVSMQINHYQYLLLLFLHESLILLSENLRKDVEAVTGSPASQTSICIGILLRSAELALLLHPVDQANTLKSPVSESVSPVVPDYLPTENGDFLSSKRKQISRDINRIRSVTVNHMSDNRSMSVDLSHIPLKDPLLFKSASDTNLQKGISFMDYLSDKHLGKISEDESSGLVYKSGSGEIGSETSDKKDSFYTDSSSILNYREDSNILSFDSDGNQNILSSTLTSKGNETIESIFKAEDLLPEAASLSENLDISKEETPPVRTLKSQSSLSGKPKERCPPNLAPLCVSYKNMKRSSSQMSLDTISLDSMILEEQLLESDGSDSHMFLEKGNKKNSTTNYRGTAESVNAGANLQNYGETSPDAISTNSEGAQENHDDLMSVVVFKITGVNGEIDIRGEDTEICLQVNQVTPDQLGNISLRHYLCNRPVGSDQKAVIHSKSSPEISLRFESGPGAVIHSLLAEKNGFLQCHIENFSTEFLTSSLMNIQHFLEDETVATVMPMKIQVSNTKINLKDDSPRSSTVSLEPAPVTVHIDHLVVERSDDGSFHIRDSHMLNTGNDLKENVKSDSVLLTSGKYDLKKQRSVTQATQTSPGVPWPSQSANFPEFSFDFTREQLMEENESLKQELAKAKMALAEAHLEKDALLHHIKKMTVE"
    seq03 = "MVAEVCSMPAASAVKKPFDLRSKMGKWCHHRFPCCRGSGKSNMGTSGDHDDSFMKTLRSKMGKCCHHCFPCCRGSGTSNVGTSGDHDNSFMKTLRSKMGKWCCHCFPCCRGSGKSNVGTWGDYDDSAFMEPRYHVRREDLDKLHRAAWWGKVPRKDLIVMLRDTDMNKRDKQKRTALHLASANGNSEVVQLLLDRRCQLNVLDNKKRTALIKAVQCQEDECVLMLLEHGADGNIQDEYGNTALHYAIYNEDKLMAKALLLYGADIESKNKCGLTPLLLGVHEQKQQVVKFLIKKKANLNALDRYGRTALILAVCCGSASIVNLLLEQNVDVSSQDLSGQTAREYAVSSHHHVICELLSDYKEKQMLKISSENSNPEQDLKLTSEEESQRLKVSENSQPEKMSQEPEINKDCDREVEEEIKKHGSNPVGLPENLTNGASAGNGDDGLIPQRKSRKPENQQFPDTENEEYHSDEQNDTQKQLSEEQNTGISQDEILTNKQKQIEVAEKEMNSKLSLSHKKEEDLLRENSMLREEIAMLRLELDETKHQNQLRENKILEEIESVKEKLLKAIQLNEEALTKTSI"
    res = compute_diversity(
        sequences=[seq01, seq02, seq03],
        mmseqs_path="/home/jhkuang/app/mmseqs/bin/mmseqs",
    )
    print(res)


def test_example_02():
    print("Test Example 02 -> Three natural sequences from ESM3.")
    seq01 = "MLLILLLLLLLLLFILFVVLLLLLLLLLLLLLLLLLLLLLLLLLLLFILLLLLLLLLSILLLLLLLLLLLLFLLLLFLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLFVVLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLFVLTFFDFALLLLLLLLLLLLLFGLLLLLLLLLFLLLLLLLLLLLLLLVLVFLLLLLLLFVVAFFFFLLSLLLLLLSIVLLFILLLFILPFILLFILLFILLLFILLLFILLLLLFILLLLFIIKKKKKKKKK"
    seq02 = "MRRAALLLLLAGLLALAAAGAAAGSPAARLAAAALALNLTPAQEAAARAALAAAAADVAAARAGDAAAAKRLAAAAARLGAPPGAAAATALRDRAAALTAAGRLAEAAAELDRALALDRLAGRPTAAVLLAQAELALARGDRAAAEAQLRAALAAADADRDAAARADALAALAARLRAAGRPDEALAARRRELAIRQRAGDVLNLPALHEAIADDFAALGRADQALAERRRAAAALAALGRDDAALAAAARALAAAAAAARDAQARARALLELARAAALAGDRAAALAALDRAEAAAAAAGDAGTAAAARLARLELLAAAGGDLAEAL"
    seq03 = "MTAAALLAAALAGALALVLAIAAGALALTLAALPAAAAKAAAALAAALALALLAAALLLAARVSAALAAAAAAAAASAAAAAAGAAAAAAAAAAAALAAALAARAAAAAAAAAAARAAAAALAPRLAALAAARAALAAERARLQARLAAAPGDAALAAALDALDRAIEALDAAEAALAAAAAAAAAAAAAAAAARAALAPALAALPALAAALAAAAAALAAALAALAAGAPPAALAAAAARLEAAAAALADEAAAPAAALARLAREGLPALAALAAVPAALAAALAAAAATAAAALAAAAAAAAALAAGGAAALLAATAAAAAAAPAA"
    res = compute_diversity(
        sequences=[seq01, seq02, seq03],
        mmseqs_path="/home/jhkuang/app/mmseqs/bin/mmseqs",
    )
    print(res)


def test():
    test_example_01()
    test_example_02()


if __name__ == "__main__":
    DEBUG = False
    if DEBUG:
        print("Debuging Diversity.")
        test()
    else:
        import fire

        fire.Fire(main)
