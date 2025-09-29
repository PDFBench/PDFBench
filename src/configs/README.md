### Configuration for PDFBench

Check the brief help with `python -m src.eval -h`.

#### Basic args `basic.*`

| Arguments           | Type          | Default   | Help Text                                                                                                                        |
| ------------------- | ------------- | --------- | -------------------------------------------------------------------------------------------------------------------------------- |
| `input_path`        | `str`         | -         | Path to the input `.json` file containing model design results.                                                                  |
| `output_dir`        | `int`         | -         | Directory for storing evaluation outputs.                                                                                        |
| `design_batch_size` | `int`         | -         | Designed sequence for every instruction in your results                                                                          |
| `dataset_type`      | `DatasetType` | -         | Input format specification for function–sequence data. PDFBench supports two built-in types and a custom type (see `README.md`). |
| `log_dir`           | `str`         | "logs"    | Directory for saving log files.                                                                                                  |
| `visualize`         | `bool`        | True      | Whether to generate a summary of evaluation results across all metrics.                                                          |
| `visual_name`       | `str`         | "results" | File name for the visualization output.                                                                                          |
| `num_gpu`           | `int`         | -1        | Number of GPUs to be used.                                                                                                       |
| `num_cpu`           | `int`         | -1        | Number of CPU cores to be used.                                                                                                  |
| `pdfbench_handler`  | `str`         | "python"  | Path to the Python interpreter for `PDFBench`. Use the default if the PDFBench environment is already activated.                 |
| `deepgo_handler`    | `str`         | -         | Path to the Python interpreter for `PDF-DeepGO`. Required when using DeepGO-SE.                                                  |

---

#### Plausibility args

##### Perplexity args `perplexity.*`

| Arguments                 | Type                          | Default                                 | Help Text                                                                                   |
| ------------------------- | ----------------------------- | --------------------------------------- | ------------------------------------------------------------------------------------------- |
| `run`                     | `bool`                        | `True`                                  | Whether to execute the perplexity evaluation.                                               |
| `name`                    | `str`                         | `"perplexity"`                          | File name for the output results.                                                           |
| `compute_models`          | `tuple[PerplexityModel, ...]` | `(ProGen2, ProtGPT2, RITA, ProteinGLM)` | Models used for perplexity computation.                                                     |
| `batch_size`              | `int`                         | `64`                                    | Batch size for perplexity computation.                                                      |
| `progen2_name_or_path`    | `str`                         | `"hugohrban/progen2-base"`              | Model name or path for ProGen2.                                                             |
| `protgpt2_name_or_path`   | `str`                         | `"nferruz/ProtGPT2"`                    | Model name or path for ProtGPT2.                                                            |
| `rita_name_or_path`       | `str`                         | `"lightonai/RITA_xl"`                   | Model name or path for RITA.                                                                |
| `proteinglm_name_or_path` | `str`                         | `"biomap-research/proteinglm-3b-clm"`   | Model name or path for ProteinGLM. Note: perplexity values from ProteinGLM may be unstable. |

##### Repeat args `repeat.*`

| Arguments         | Type                           | Default            | Help Text                                         |
| ----------------- | ------------------------------ | ------------------ | ------------------------------------------------- |
| `run`             | `bool`                         | `True`             | Whether to execute the repetitiveness evaluation. |
| `name`            | `str`                          | `"repetitiveness"` | File name for the output results.                 |
| `compute_methods` | `tuple[Repeat_Algorithm, ...]` | `(Repeat, RepN)`   | Algorithms used to evaluate repetitiveness.       |
| `RepN`            | `tuple[int, ...]`              | `(2, 5)`           | Values of N for the RepN method.                  |

---

#### Foldability args `foldability.*`

| Arguments               | Type   | Default                 | Help Text                                      |
| ----------------------- | ------ | ----------------------- | ---------------------------------------------- |
| `run`                   | `bool` | `True`                  | Whether to execute the foldability evaluation. |
| `name`                  | `str`  | `"foldability"`         | File name for the output results.              |
| `pdb_cache_dir`         | `str`  | `"pdb_cache_dir/"`      | Directory for PDB cache files.                 |
| `esm_fold_name_or_path` | `str`  | `"facebook/esmfold_v1"` | Model name or path for ESMFold weights.        |

---

#### Language Alignment args

##### ProTrek Score args `protrek_score.*`

| Arguments      | Type          | Default           | Help Text                                        |
| -------------- | ------------- | ----------------- | ------------------------------------------------ |
| `protrek_path` | `str \| None` | -                 | Path to ProTrek-650M weights.                    |
| `run`          | `bool`        | `True`            | Whether to execute the ProTrek Score evaluation. |
| `name`         | `str`         | `"protrek_score"` | File name for the output results.                |

##### EvoLlama Score args `evollama_score.*`

| Arguments                 | Type   | Default                              | Help Text                                           |
| ------------------------- | ------ | ------------------------------------ | --------------------------------------------------- |
| `evollama_path`           | `str`  | -                                    | Path to `EvoLlama Score` weights.                   |
| `llama_name_or_path`      | `str`  | `"meta-llama/Llama-3.2-3B-Instruct"` | Model name or path for `Llama-3.2-3B-Instruct`.     |
| `pubmedbert_name_or_path` | `str`  | `"NeuML/pubmedbert-base-embeddings"` | Model name or path for `PubMedBERT embeddings`.     |
| `run`                     | `bool` | `True`                               | Whether to execute the `EvoLlama Score` evaluation. |
| `name`                    | `str`  | `"evollama_score"`                   | File name for the output results.                   |

##### GO Recovery `go_score.*`

| Arguments            | Type    | Default      | Help Text                                      |
| -------------------- | ------- | ------------ | ---------------------------------------------- |
| `deepgo_weight_path` | `str`   | -            | Path to DeepGO-SE weights.                     |
| `deepgo_threshold`   | `float` | `0.7`        | Confidence threshold for GO prediction.        |
| `deepgo_batch_size`  | `int`   | `64`         | Batch size for DeepGO-SE.                      |
| `run`                | `bool`  | `True`       | Whether to execute the GO Recovery evaluation. |
| `name`               | `str`   | `"go_score"` | File name for the output results.              |

##### IPR Recovery `ipr_score.*`

| Arguments               | Type   | Default       | Help Text                                       |
| ----------------------- | ------ | ------------- | ----------------------------------------------- |
| `interpro_scan_ex_path` | `str`  | -             | Path to the InterProScan executable.            |
| `interpro_cache_path`   | `str`  | -             | Path to the InterProScan cache directory.       |
| `workers_per_scan`      | `int`  | -1            | Number of workers used for InterProScan.        |
| `run`                   | `bool` | `True`        | Whether to execute the IPR Recovery evaluation. |
| `name`                  | `str`  | `"ipr_score"` | File name for the output results.               |

##### Retrieval Accuracy `retrieval_acc.*`

| Arguments                | Type                              | Default                | Help Text                                             |
| ------------------------ | --------------------------------- | ---------------------- | ----------------------------------------------------- |
| `protrek_path`           | `str`                             | `None`                 | Path to ProTrek Score weights.                        |
| `protrek_batch_size`     | `int`                             | `None`                 | Batch size for ProTrek evaluation.                    |
| `retrieval_difficulties` | `tuple[RetrievalDifficulty, ...]` | `(Soft, Normal, Hard)` | Difficulty levels for retrieval accuracy evaluation.  |
| `run`                    | `bool`                            | `True`                 | Whether to execute the Retrieval Accuracy evaluation. |
| `name`                   | `str`                             | `"retrieval_accuracy"` | File name for the output results.                     |

---

#### Novelty `novelty.*`

| Arguments                | Type                  | Default               | Help Text                                  |
| ------------------------ | --------------------- | --------------------- | ------------------------------------------ |
| `mmseqs_ex_path`         | `str`                 | -                     | Path to the MMseqs executable.             |
| `foldseek_ex_path`       | `str`                 | -                     | Path to the Foldseek executable.           |
| `mmseqs_targetdb_path`   | `str`                 | -                     | Path to the MMseqs search database.        |
| `foldseek_targetdb_path` | `str`                 | -                     | Path to the Foldseek search database.      |
| `run`                    | `bool`                | `True`                | Whether to execute the Novelty evaluation. |
| `name`                   | `str`                 | `"novelty"`           | File name for the output results.          |
| `workers_per_mmseqs`     | `int`                 | -1                    | Number of workers used for MMseqs.         |
| `workers_per_foldseek`   | `int`                 | -1                    | Number of workers used for Foldseek.       |
| `novelties`              | `tuple[Novelty, ...]` | `(Novelty.Sequence,)` | Types of novelty considered.               |
| `pdb_cache_dir`          | `str`                 | `"pdb_cache_dir/"`    | Directory for PDB cache files.             |

---

#### Diversity `diversity.*`

| Arguments               | Type                    | Default                 | Help Text                                    |
| ----------------------- | ----------------------- | ----------------------- | -------------------------------------------- |
| `mmseqs_ex_path`        | `str`                   | -                       | Path to the MMseqs executable.               |
| `tm_score_ex_path`      | `str`                   | -                       | Path to the TMScore executable.              |
| `esm_fold_name_or_path` | `str`                   | `"facebook/esmfold_v1"` | Model name or path for ESMFold.              |
| `pdb_cache_dir`         | `str`                   | `"pdb_cache_dir/"`      | Directory for PDB cache files.               |
| `diversities`           | `tuple[Diversity, ...]` | `(Sequence, Structure)` | Types of diversity considered.               |
| `run`                   | `bool`                  | `True`                  | Whether to execute the Diversity evaluation. |
| `name`                  | `str`                   | `"diversity"`           | File name for the output results.            |

---

#### Similarity

##### GT-Identity `identity.*`

| Arguments           | Type   | Default      | Help Text                                      |
| ------------------- | ------ | ------------ | ---------------------------------------------- |
| `run`               | `bool` | `True`       | Whether to execute the GT-Identity evaluation. |
| `name`              | `str`  | `"identity"` | File name for the output results.              |
| `thread_per_mmseqs` | `int`  | `6`          | Number of threads used per MMseqs run.         |
| `mmseqs_ex_path`    | `str`  | -            | Path to the MMseqs executable.                 |

##### GT-TMscore `tm_score.*`

| Arguments               | Type   | Default                 | Help Text                                     |
| ----------------------- | ------ | ----------------------- | --------------------------------------------- |
| `run`                   | `bool` | `True`                  | Whether to execute the GT-TMScore evaluation. |
| `name`                  | `str`  | `"tm_score"`            | File name for the output results.             |
| `tm_score_ex_path`      | `str`  | -                       | Path to the TMScore executable.               |
| `esm_fold_name_or_path` | `str`  | `"facebook/esmfold_v1"` | Model name or path for ESMFold.               |
| `pdb_cache_dir`         | `str`  | `"pdb_cache_dir/"`      | Directory for PDB cache files.                |

##### ESMScore `bert_score.*`

| Arguments           | Type                    | Default                          | Help Text                                   |
| ------------------- | ----------------------- | -------------------------------- | ------------------------------------------- |
| `run`               | `bool`                  | `True`                           | Whether to execute the ESMScore evaluation. |
| `name`              | `str`                   | `"bert_score"`                   | File name for the output results.           |
| `compute_models`    | `tuple[BertModel, ...]` | `(ESM2,)`                        | Models used to compute ESMScore.            |
| `esm2_name_or_path` | `str`                   | `"facebook/esm2_t33_650M_UR50D"` | Model name or path for ESM2.                |
| `esm2_batch_size`   | `int`                   | `32`                             | Batch size for ESM2.                        |
