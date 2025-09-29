## Repository for PDFBench: A Benchmark for De novo Protein Design from Function
The paper can be viewed on the homepage: https://pdfbench.github.io/

### 1. Environment
```shell
# PDFBench env
conda create -n PDF python=3.10
conda activate PDF
bash scripts/environments/PDF.sh    # install the PDFBench requirements

# DeepGo env
conda activate -n PDF-DeepGO python-3.10
conda activate PDF-DeepGO
bash scripts/environments/DeepGO.sh # install the DeepGO-SE requirements
```

### 2. Repository Structure
| Dictionary | Usage                                                                            |
| ------------ | ------------------------------------------------------------------------------ |
| data         | Two test sets (SwissTest and MolinstTest) and two design results are included. |
| example      | Two examples for description-guided and keyword-guided task.                   |
| scripts      | Scripts for environment preparation                                            |
| tools        | Dictionary for downloading the third-party tools.                              |
| weights      | Dictionary for downloading the model weights.                                  |

We provide more detailed introductions in these directories.

### 3. Preparation

In this section, we outline the configuration process for the environment required by PDFBench. 
> For download paths of software tools such as `MMseqs2` and `Foldseek`, as well as model weights, please refer to the provided resources or specify them in the final configuration file.

#### 3.1. ProTrek and EvoLlama

Download the [ProTrek-650M weights](https://huggingface.co/westlake-repl/ProTrek_650M_UniRef50) and [EvoLlama weights](https://huggingface.co/nwliu/EvoLlama-Oracle-Molinst-Protein-Design) to folder `weights` following their guidelines, we here provide the easy-go guidance,

```shell
cd weights
huggingface-cli download westlake-repl/ProTrek_650M \
                         --repo-type model \
                         --local-dir ./ProTrek_650M
huggingface-cli download nwliu/EvoLlama \
                         --repo-type model \
                         --local-dir ./EvoLlama
```

The model weights for *ProTrek_650M* and *EvoLlama* are located at `weights/ProTrek_650M` and `weights/EvoLlama`, respectively. These paths can be assigned to `protrek_score.protrek_path` and `evollama_score.evollama_path` in the configuration file.

#### 3.2. TMscore
Download TMscore following the [ZhangGroup](https://zhanggroup.org/TM-score/). According to the guidance, your directory may look like:
```shell
cd tools/TMScore/

wget http://zhanggroup.org/TM-score/TMscore.cpp
g++ -O3 -o TMscore TMscore.cpp

tree . -L 1
├── TMscore
└── TMscore.cpp
```
The path to *TMscore* executable file is `tools/TMScore/TMscore`. The path can be assigned to `tm_score.tm_score_ex_path`. 

#### 3.3. InterProScan
InterProScan needs Java11!
```shell
cd tools/InterProScan
wget https://ftp.ebi.ac.uk/pub/software/unix/iprscan/5/5.75-106.0/interproscan-5.75-106.0-64-bit.tar.gz
tar -zxvf ./interproscan-5.75-106.0-64-bit.tar.gz
```
The path to *InterProScan* executable file is `tools/InterProScan/interproscan/interproscan-5.75-106.0-64/interproscan.sh`, which can be assigned to `ipr_score.interpro_scan_ex_path`.

#### 3.4. MMseqs2 and its database
Download MMseqs2 following the [tutorial](https://github.com/soedinglab/MMseqs2).
```shell
cd tools/MMseqs
wget https://mmseqs.com/latest/mmseqs-linux-gpu.tar.gz
tar xvfz mmseqs-linux-gpu.tar.gz
export PATH=$(pwd)/mmseqs/bin/:$PATH

cd mmseqs
tree . -L 2
├── bin
│   └── mmseqs  # Executable file of MMSeqs2
├── examples
├── LICENSE.md
├── matrices
├── README.md
├── userguide.pdf
└── util
```

The path to executable file of *MMSeqs2* is `tools/MMseqs/mmseqs/bin/mmseqs`, which can be assigned to `novelty.mmseqs_ex_path` in the configuration file.

```shell
cd tools/MMseqs
mkdir db && cd db

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
The path to *MMSeqs DB* is `tools/MMseqs/db/uniprotdb` (CPU version) or `tools/MMseqs/db/uniprotdb_gpu` (GPU version), which can be set to `novelty.mmseqs_targetdb_path`.

**Warning:** The search database constructed from UniProtKB occupies approximately 500 GB of disk space and requires tens of hours to build. Additionally, searching a single sequence without GPU acceleration may take nearly one hour to complete.

#### 3.5. Foldseek and its database
Download Foldseek following the [tutorial](https://github.com/steineggerlab/foldseek).

```shell
cd tools/Foldseek
wget https://mmseqs.com/foldseek/foldseek-linux-gpu.tar.gz
tar xvfz foldseek-linux-gpu.tar.gz
export PATH=$(pwd)/foldseek/bin/:$PATH

cd foldseek
tree .
├── bin
│   └── foldseek
└── README.md
```

The path to executable file of *Foldseek* is `tools/Foldseek/foldseek/bin/foldseek`, which can be assigned to `novelty.foldseek_ex_path` in the configuration file.

```shell
cd tools/Foldseek
mkdir db && cd db

# Downloading AlphaFoldDB/SwissProt
wget https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/swissprot_pdb_v4.tar
tar -xvf swissprot_pdb_v4.tar

# Creating searching db of CPU version.
mkdir AlphaSwissCPU
foldseek createdb swissprot_pdb_v4 ./AlphaSwissCPU/AlphaC
# Converting searching db to GPU version.
mkdir AlphaSwissGPU
foldseek makepaddedseqdb ./AlphaSwissCPU/AlphaC ./AlphaSwissGPU/AlphaG
# Craeating indexes for searching acceleration. (tens of hours)
mmseqs createindex ./AlphaSwissGPU/AlphaG ./tmp --index-subset 2
```
The path to *Foldssek DB* is `tools/MMseqs/db/AlphaSwissCPU/AlphaC` (CPU version) or `tools/MMseqs/db/AlphaSwissGPU/AlphaG` (GPU version), which can be set to `novelty. foldseek_targetdb_path`.

**Note:** The search database constructed from AlphaFoldDB/SwissProt occupies approximately 27 GB of disk space and requires several hours to build. Structure searches without GPU acceleration may take tens of minutes per query. Due to the substantial storage requirements of the full PDB database (~1TB), we ultimately adopted AlphaFoldDB/SwissProt as a practical compromise.

#### 3.7. Download weights of DeepGO-SE
Following the guidance of DeepGO-SE, download the weights,

```shell
cd weights/DeepGO

wget https://bio2vec.net/data/deepgo/deepgo2/data.tar.gz
tar xvzf data.tar.gz
```
The path to DeepGO-SE weights is `weights/DeepGO/data`, which can be set to `go_score.deepgo_weights_path`.

### 4. Prepare your evaluation data

#### 4.1. Reorganize your data

##### Description-guided Task
For Description-guided task, we highly recommend that you organize evaluation data like us, see `data/designed/description/ProDVa.json`.
- `instruction`: Protein functions described in natural language.
- `reference`: *Ground Truth* for design sequence.
- `response#N`: N-th design sequence.

##### Keyword-guided Task
For Keyword-guided task, you need provide Gene Ontology terms or InterPro entries for every designed sequence, see `data/designed/keyword/CFP-Gen.json`.
- `Gene Ontology (molecular function)`: Gene Ontology terms list for designed sequence.
    - `GO-ID`: name for Gene Ontology term.
    - `GO-Name`: name for the Gene Ontology term.
- `InterPro`: InterPro entries list for designed sequence.
    - `InterPro-ID`: name for InterPro entry.
    - `InterPro-name`: name for the InterPro entry.
    - `InterPro-Type`: type for the InterPro entry.
    - `Beg`, `End`: The start and end positions on the sequence for this entry are provided for reference, **not required for the PDFBench evaluation**.
- `reference`: *Ground Truth* for design sequence.
- `response#N`: N-th design sequence.s

##### Keyword-Description-guided Task
Support for Keyword-Description-guided task will come soon.

#### 4.2. Customize your function parser

See `src/datasets`, We provide two parsers for the description-guided and keyword-guided tasks, both as defined in PDFBench. These parsers are designed to extract subsequences that directly describe protein function, while minimizing the impact of manual prompts on computational language alignment. The two parsers are displayed as follows.

```python
# Separate the instruction and the function for better display, 
# they are all essentially continuous textual descriptions.
"""
Keyword-guided Task
"""
instruction: str = (
    "Generate a protein sequence for a novel protein that integrates the following "
    "function keywords: Cyt_c-like_dom. The designed protein sequence is "
)
class KeywordDataset(BaseDataset):
    @classmethod
    def function(cls, instruction: str) -> str:
        keyword = instruction.removesuffix(
            "The designed protein sequence is "
        )
        keyword = re.search(r":\s*(.*)", keyword[:-2]).group(1)
        return keyword.strip()
parsed_function: str = Description.function(instruction)
print(parsed_function)
# 'Cyt_c-like_dom'

"""
Description-guided Task
"""
instruction: str = ( 
    "Develop a protein sequence with increased stability under specific condition. "
    "1. Mn(2+) is required for the proper folding and stability of the protein. "
    "2. The protein design should emphasize Polar residues, Basic and acidic residues compositional bias. "
    "3. The designed protein must have at least one Nudix hydrolase domain. "
    "The designed protein sequence is "
)
class DescriptionDataset(BaseDataset):
    @classmethod
    def function(cls, instruction: str) -> str:
            function = re.sub(r"^.*?(1\.)", r"\1", instruction)
            function = function.removesuffix(
                "The designed protein sequence is "
            )
            return function.strip()
parsed_function: str = Description.function(source)
print(parsed_function)
# 1. Mn(2+) is required for the proper folding and stability of the protein. 
# 2. The protein design should emphasize Polar residues, Basic and acidic residues compositional bias. 
# 3. The designed protein must have at least one Nudix hydrolase domain.
```

While PDFBench work on your design results, not the model directly, instruction format for your model may differ from the pre-set one. Therefore, we provide a costumed dataset class for the costumed instruction. You can modify the method `function` of the class `CostumedDataset` in `src/datasets/custom.py`, and set `basic.dataset_type`  to `CostumedDataset`or pass the parameter `--basic.dataset_type CostumedDataset` when execute the PDFBench.

**Note:** this is very important for metric using actual protein function, namely *ProTrek Score*, *EvoLlama Score*, *Retrieval Accuracy*, for the fair evaluation.

### 5. Setting your configuration.
The final step before running the benchmark: configure the PDFBench. You can create a yml file, with reference to the two examples `example/configs/description-guided/ProDVa.yml` and `example/configs/keyword-guided/ipr-go/CFP-Gen.yml`.

<!-- | Argument name | Usage |
| ------------- | ----- |
|               |       |
|               |       |
|               |       | -->

### 6. Let's Go Evaluation!
```shell
conda activate PDFBench
python -m src.eval --config path/to/config/yml
```

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