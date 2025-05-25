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

Download TMscore following the [ZhangGroup](https://zhanggroup.org/TM-score/).

#### 2.3. InterProScan

Download InterProScan for Keyword Recovery.
```shell
cd path/to/interproscan
wget http://ftp.ebi.ac.uk/pub/software/unix/iprscan/5/5.74.105/interproscan-5.74.105-64-bit.tar.gz
tar -zxvf ./interproscan-5.74.105-64-bit.tar.gz
```
The path to InterProScan executable file is `path/to/interproscan/interproscan-5.74-105.0/interproscan.sh`. 

#### 2.4.MMseqs2 and its database

Download MMseqs2 following the [tutorial](https://github.com/soedinglab/MMseqs2).
```shell
cd path/to/mmseqs/DB

wget https://ftp.uniprot.org/pub/databases/uniprot/knowledgebase/complete/uniprot_sprot.fasta.gz
gunzip uniprot_sprot.fasta.gz
wget https://ftp.uniprot.org/pub/databases/uniprot/knowledgebase/complete/uniprot_trembl.fasta.gz
gunzip uniprot_trembl.fasta.gz

cat uniprot_sprot.fasta uniprot_trembl.fasta > uniprot.fasta

mmseqs createdb uniprot.fasta ./uniprotdb
mmseqs createindex ./uniprotdb tmp
```
The path to MMSeqs2 seaching DB is `path/to/mmseqs/DB/uniprotdb`

### 3. prepare your evaluation data

we highly recommand that you organize evaluation data like [us](./example/data/example_data.json).
- instruction: Protein functions described in natural language
- reference: Ground Truth sequence
- response: Designed sequence

### 4. Let's Go Evaluation!

Edit your settings above in `path/to/PDFBench/scripts/eval.sh`.
```shell
cd path/to/PDFBench
zsh scripts/eval.sh
```