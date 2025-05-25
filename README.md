## Repository for PDFBench: A Benchmark for De novo Protein Design from Function

### 1. Environment
```shell
conda creat -n PDF --file reqiurements.txt
conda activate PDF
```

### 2. Preparsion

#### 2.1. ProTrek and EvoLlama

Download repository for both [ProTrek](https://github.com/westlake-repl/ProTrek) and [EvoLlama](https://github.com/sornkL/EvoLlama) into `src` folder, and download the [ProTrek-650M weights](https://huggingface.co/westlake-repl/ProTrek_650M_UniRef50) and [EvoLlama weights](https://huggingface.co/nwliu/EvoLlama-Oracle-Molinst-Protein-Design) following their guidelines.

#### 2.2. TMscore
Download TMscore following the [ZhangGroup](https://zhanggroup.org/TM-score/). According to the guidance, your directionary may look like:
```shell
cd /path/to/TMscore
tree . -L 1
├── TMscore # Execuatable file of TMscore, it will be used later.
└── TMscore.cpp
```
The path to **TMscore** executable file is `/path/to/TMscore/TMscore`. 

#### 2.3. InterProScan
```shell
cd /path/to/interproscan
wget http://ftp.ebi.ac.uk/pub/software/unix/iprscan/5/5.74.105/interproscan-5.74.105-64-bit.tar.gz
tar -zxvf ./interproscan-5.74.105-64-bit.tar.gz
```
The path to **InterProScan** executable file is `/path/to/interproscan/interproscan-5.73-104.0/interproscan.sh`. 

#### 2.4. MMseqs2 and its database
Download MMseqs2 following the [tutorial](https://github.com/soedinglab/MMseqs2).According to the guidance, your directionary may look like:
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
The path to execuatable file of **MMSeqs2** is `/path/to/mmseqs/bin/mmseqs`.
```shell
cd /path/to/mmseqs
mkdir DB && cd DB

# Downloading UniProtKB/SwissProt (~400M)
wget https://ftp.uniprot.org/pub/databases/uniprot/knowledgebase/complete/uniprot_sprot.fasta.gz
gunzip uniprot_sprot.fasta.gz
# Downloading UniProtKB/Trembl (~100G)
wget https://ftp.uniprot.org/pub/databases/uniprot/knowledgebase/complete/uniprot_trembl.fasta.gz
gunzip uniprot_trembl.fasta.gz

# Concatenanting two subsets to UniProtKBs
cat uniprot_sprot.fasta uniprot_trembl.fasta > uniprot.fasta

# Creating searching db of CPU version.
mmseqs createdb uniprot.fasta ./uniprotdb
# Converting searching db to GPU version.
mmseqs makepaddedseqdb ./uniprotdb ./uniprotdb_gpu
# Craeating Indexes for searching acceleration. (tens of hours)
mmseqs createindex ./uniprotdb_gpu tmp --index-subset 2
```
The path to **MMSeqs DB** is `path/to/mmseqs/DB/uniprotdb` or `path/to/mmseqs/DB/uniprotdb_gpu`.
**Warning:** The searching DB bulided from UniProtKB takes up about **500 GB** of disk space and runs for tens of hours, and it takes nearly **1 hour** to complete the searching of one single sequence if no GPU acceleration!

### 3. prepare your evaluation data

we highly recommand that you organize evaluation data like [us](./example/data/example_data.json).
- `instruction`: Protein functions described in natural language
- `reference`: Ground Truth sequence
- `response`: Designed sequence
### 4. Let's Go Evaluation!
Edit your preparesion in `./scripts/eval.sh`.
```shell
zsh scripts/eval.sh
```