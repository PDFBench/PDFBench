## Repository for PDFBench: A Benchmark for De novo Protein Design from Function
The paper can be viewed on the homepage: https://pdfbench.github.io/

### 1. Environment
```shell
conda create -n PDF python=3.10
conda activate PDF

conda install pytorch::faiss-gpu=1.8.0 --yes

pip install torch
pip install transformers
pip install accelerate
pip install esm
pip install simple_parsing

pip install httpx
pip install torchmetrics==0.9.3
pip install pytorch-lightning==2.1.3
```

### 2. Preparation

#### 2.1. ProTrek and EvoLlama

Download repository for both [ProTrek](https://github.com/westlake-repl/ProTrek) and [EvoLlama](https://github.com/sornkL/EvoLlama) into `src` folder, and download the [ProTrek-650M weights](https://huggingface.co/westlake-repl/ProTrek_650M_UniRef50) and [EvoLlama weights](https://huggingface.co/nwliu/EvoLlama-Oracle-Molinst-Protein-Design) following their guidelines.

#### 2.2. TMscore
Download TMscore following the [ZhangGroup](https://zhanggroup.org/TM-score/). According to the guidance, your directory may look like:
```shell
cd /path/to/TMscore
tree . -L 1
├── TMscore # Executable file of TMscore, it will be used later.
└── TMscore.cpp
```
The path to **TMscore** executable file is `/path/to/TMscore/TMscore`. 

#### 2.3. InterProScan
InterProScan needs Java11!
```shell
cd /path/to/interproscan
wget http://ftp.ebi.ac.uk/pub/software/unix/iprscan/5/5.74.105/interproscan-5.74.105-64-bit.tar.gz
tar -zxvf ./interproscan-5.74.105-64-bit.tar.gz
```
The path to **InterProScan** executable file is `/path/to/interproscan/interproscan-5.73-105-64/interproscan.sh`. 

#### 2.4. MMseqs2 and its database
Download MMseqs2 following the [tutorial](https://github.com/soedinglab/MMseqs2).According to the guidance, your directory may look like:
```shell
cd /path/to/mmseqs
tree . -L 2
├── bin
│   └── mmseqs  # Execuatable file of MMSeqs2
├── examples
├── LICENSE.md
├── matrices
├── README.md
├── userguide.pdf
└── util
```
The path to executable file of **MMSeqs2** is `/path/to/mmseqs/bin/mmseqs`.
```shell
cd /path/to/mmseqs
mkdir DB && cd DB

# Downloading UniProtKB/SwissProt (~400M)
wget https://ftp.uniprot.org/pub/databases/uniprot/knowledgebase/complete/uniprot_sprot.fasta.gz
gunzip uniprot_sprot.fasta.gz
# Downloading UniProtKB/Trembl (~100G)
wget https://ftp.uniprot.org/pub/databases/uniprot/knowledgebase/complete/uniprot_trembl.fasta.gz
gunzip uniprot_trembl.fasta.gz

# Concatenating two subsets to UniProtKBs
cat uniprot_sprot.fasta uniprot_trembl.fasta > uniprot.fasta

# Creating searching db of CPU version.
mmseqs createdb uniprot.fasta ./uniprotdb
# Converting searching db to GPU version.
mmseqs makepaddedseqdb ./uniprotdb ./uniprotdb_gpu
# Craeating indexes for searching acceleration. (tens of hours)
mmseqs createindex ./uniprotdb_gpu tmp --index-subset 2
```
The path to **MMSeqs DB** is `/path/to/mmseqs/DB/uniprotdb` or `/path/to/mmseqs/DB/uniprotdb_gpu`.
**Warning:** The searching DB bulit from UniProtKB takes up about **500 GB** of disk space and runs for tens of hours, and it takes nearly **1 hour** to complete the searching of one single sequence if no GPU acceleration!

#### 2.5. ESMFold
Download ESMFold from [huggingface](https://huggingface.co/facebook/esmfold_v1), and the path to ESMFold weights is `/path/to/esmfold/weights/folder`

#### 2.6. Modify your function-parser
See `src/utils.py`, we provide two parsers for Mol-Instructions and CAMEOTest as follows,
```python
"""
Keyword-guided Task
"""
source: str = "Generate a protein sequence for a novel protein that integrates the following function keywords: Cyt_c-like_dom. The designed protein sequence is "
def get_text_from_keywords(instruction: str) -> str:
    # Function for parsing keywords from text
    keyword = instruction.removesuffix("The designed protein sequence is ")
    keyword = re.search(r":\s*(.*)", keyword[:-2]).group(1)
    return keyword.strip()
keywords: str = get_text_from_keywords(source)  
# Keywords 'Cyt_c-like_dom' left only

"""
Description-guided Task
"""
source: str = "Synthesize a protein sequence with the appropriate folding and stability properties for the desired function. 1. The protein should be able to modulate glycine decarboxylation via glycine cleavage system in a way that leads to a desirable outcome. The designed protein sequence is "
def get_text_from_description(instruction: str) -> str:
    # Function for parse description from text
    function = re.sub(r"^.*?(1\.)", r"\1", instruction)
    function = function.removesuffix("The designed protein sequence is ")
    return function.strip()
description: str = get_text_from_description(source)
# Additional prompt 'The designed protein sequence is ' is deleted.
```
If your `Function` description/keyword differs, you must modify these two functions to get coorect performance in `ProTrek Score`, `Evollama Score` and `Retrieval Accuracy`.

### 3. Prepare your evaluation data
we highly recommend that you organize evaluation data like us, see `./example/data/example_data.json`
- `instruction`: Protein functions described in natural language
- `reference`: Ground Truth protein sequence
- `response`: Designed protein sequence

### 4. Let's Go Evaluation!
We Provide two examples for single and batch evaluation. You may edit your preparation in `./scripts/eval.sh` following them.
```shell
zsh scripts/eval.sh
```
Note: we provide example result files, which should be deleted initially.

### Cite this work
```bibtex
@misc{kuang2025pdfbenchbenchmarknovoprotein,
      title={PDFBench: A Benchmark for De novo Protein Design from Function}, 
      author={Jiahao Kuang and Nuowei Liu and Changzhi Sun and Tao Ji and Yuanbin Wu},
      year={2025},
      eprint={2505.20346},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2505.20346}, 
}
```