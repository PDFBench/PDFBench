### Implementation for evaluation metrics
#### Plausibility
perplexity: `sequence/perplexity.py`
repetitiveness: `sequence/repetitiveness.py`
#### Foldability
foldability: `structure/foldability.py`
#### Language Alignment
ProTrek Score: `alignment/protrek_score.py`
EvoLlama Score: `alignment/evollama_score.py`
GO Recovery: `alignment/go_score.py`
IPR Recovery: `alignment/ipr_score.py`
Retrieval Accuracy: `alignment/retrieval_accuracy.py`
#### Novelty
Novelty-Seq/Struct~Easy/Hard~: `others/novelty.py`
#### Diversity
Diversity-Seq/Struct: `others/diversity.py`
#### Similarity
GT-Identity: `sequence/identity.py`
GT-TMScore: `structure/tm_score.py`
ESMScore: `sequence/bert_score.py`