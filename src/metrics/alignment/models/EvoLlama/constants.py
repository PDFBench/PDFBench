# This file is copied from [EvoLlama](https://github.com/sornkL/EvoLlama)
# Original license: MIT License
# Special tokens for the embeddings of structures
STRUCTURE_BEGIN_TOKEN = "<structure>"
STRUCTURE_END_TOKEN = "</structure>"

# Special tokens for the embeddings of sequences
SEQUENCE_BEGIN_TOKEN = "<sequence>"
SEQUENCE_END_TOKEN = "</sequence>"

# Special tokens for the embeddings of structures and sequences (fusion)
PROTEIN_BEGIN_TOKEN = "<protein>"
PROTEIN_END_TOKEN = "</protein>"

SEPERATOR = "<sep>"

# Protein structure encoder names
STRUCTURE_ENCODER_PROTEIN_MPNN = "ProteinMPNN"
STRUCTURE_ENCODER_GEARNET = "GearNet"

SYSTEM_MESSAGE = "You are a helpful AI assistant designed to understand representations of protein structures and sequences, and to assist users with a variety of tasks using natural language."

IGNORE_INDEX = -100
IGNORE_POSITION_ID = -1
