#!/bin/bash

# PDFBench requirement
conda install pytorch::faiss-gpu=1.8.0 --yes
pip install torch
pip install transformers
pip install accelerate
pip install esm
pip install simple_parsing
# requirements for sub-module [ProTrek] 
pip install httpx
pip install torchmetrics==0.9.3
pip install pytorch-lightning==2.1.3
# requirements for sub-module [EvoLlama]
pip install peft