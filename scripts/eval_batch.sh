# Scripts for evaluation of PDFBench
# This script evaluate batch designed result.

# region 1.Settings(See README)
export TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_EVALUATE_OFFLINE=1
# The path to PDFBench project
PROJECT_ROOT=/path/to/PDFBench
# The directionary containing data for evaluation
EVAL_DIR=${PROJECT_ROOT}/example/single
# The evaluation task of your data
# We support description-guided and keyword-guided
TASK=description-guided # keyword-guided task need more preparsion, see README

# region 1.1.Evaluation Settings(See README)
# ProTrek weights
PROTREK=/path/to/ProTrek-650M/weights/folder
# EvoLlama weights
EVOLLAMA=/path/to/EvoLlama/weights/folder
LLAMA=/path/to/Llama-3.2-3B-Instruct/weigths/folder
# ESMFold weights
ESMFOLD=/path/to/esmfold/weights/folder
# MMseqs2
MMSEQS_EX=/path/to/mmseqs/bin/mmseqs
MMSEQS_DB=/path/to/mmseqs/DB/uniprotdb_gpu
# Interproscan
INTERPRO_SCAN_EX=/path/to/interproscan/interproscan-5.73-104.0/interproscan.sh
# TMscore
TM_SCORE_EX=/path/to/TMscore/TMscore
# endregion 

# region 1.2.Device Settings
# NO-GPU Workers(We recommand you to utilize >= 6 workers for Retrieval-based taskes)
NUM_WORKERS=$(( $(nproc) / 6 )) 
# GPU Workers(Specify the number currently available on your machine)
export CUDA_VISIBLE_DEVICES=0,1,2,3 
# Workers for InterProScan, we strongly recommend utilizing all process of your machine to accelerate the search process.
WORKERS_PER_INTERPRO=$(( $(nproc) ))  
# Without modification
NUM_DEVICES=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
# endregion Device

# region 1.3.Batch Settings
EPOCHES=(1 2 3) # We evaluate all baseline for 3 epoches.
# endregion

# region 1.4.Setting Display
echo ">>>  $(date "+[%-m-%d-%H:%M:%S]") [Batch Settings] >>>"
echo Project Root: ${PROJECT_ROOT}
echo Evaluation Directory: ${EVAL_DIR}
echo Task: ${TASK}
echo Epoches: ${EPOCHES[@]}
echo ">>1.Tools"
echo ">ProTrek:" ${PROTREK}
echo ">EvoLlama:" ${EVOLLAMA}
echo ">Llama:" ${LLAMA}
echo ">MMSEQS_EX:" ${MMSEQS_EX}
echo ">MMSEQS_DB:" ${MMSEQS_DB}
echo ">INTERPRO_SCAN_EX:" ${INTERPRO_SCAN_EX}
echo ">TM_SCORE_EX:" ${TM_SCORE_EX}
echo ">>Devices"
echo ">NUM_DEVICES:" ${NUM_DEVICES}
echo ">CUDA_VISIBLE_DEVICES:" ${CUDA_VISIBLE_DEVICES}
echo ">NUM_WORKERS:" ${NUM_WORKERS}
echo ">WORKERS_PER_INTERPRO:" ${WORKERS_PER_INTERPRO}
# endregion

# endregion 1.Settings

# region 2.Evaluation
cd ${PROJECT_ROOT}
for EPOCH in ${EPOCHES[@]}; do
    if [ ! -d ${save_dir} ]; then   # Path check
        echo ${save_dir} does not exist >&2
        exit 1
    fi
    eval_dir=${EVAL_DIR}/${EPOCH}
    # region 2.1.GPU-based

    # Perplexity
    echo ">>> $(date "+[%-m-%d-%H:%M:%S]") [Perplexity] >>>"
    python -m src.perplexity \
    --num-workers $NUM_DEVICES \
    --sequence-file $EVAL_DIR/designed.json \
    --evaluation-file $EVAL_DIR/perplexity.json

    # Retrieval Accuracy
    echo ">>> $(date "+[%-m-%d-%H:%M:%S]") [Retrieval Accuracy] >>>"
    python -m src.retrieval_accuracy \
    --sequence-file ${EVAL_DIR}/designed.json \
    --evaluation-file ${EVAL_DIR}/retrieval_accuracy.json \
    --model_path ${PROTREK} \
    --task ${TASK} \
    --num-workers ${NUM_DEVICES}

    # Bert-like
    echo ">>> $(date "+[%-m-%d-%H:%M:%S]") [Bert-like] >>>"
    python -m src.bertlike \
    --num-workers $NUM_DEVICES \
    --sequence-file $EVAL_DIR/designed.json \
    --evaluation-file $EVAL_DIR/bertlike.json

    # Foldability
    echo ">>> $(date "+[%-m-%d-%H:%M:%S]") [Foldability] >>>"
    python -m src.foldability \
    --num-workers ${NUM_DEVICES} \
    --sequence-file ${EVAL_DIR}/designed.json \
    --evaluation-file ${EVAL_DIR}/foldability.json \
    --esmfold_path ${ESMFOLD} \
    --output-pdb-dir ${EVAL_DIR}/pdb_esmfold_v1

    # Language Alignment
    echo ">>> $(date "+[%-m-%d-%H:%M:%S]") [Language Alignment] >>>"
    python -m src.language_alignment \
    --num-workers ${NUM_DEVICES} \
    --sequence-file ${EVAL_DIR}/designed.json \
    --evaluation-file ${EVAL_DIR}/language_alignment.json \
    --task ${TASK} \
    --use-structure False \
    --use-sequence True \
    --pdb-dir ${EVAL_DIR}/pdb_esmfold_v1 \
    --protrek-path ${PROTREK} \
    --evollama-path ${EVOLLAMA} \
    --llm-path ${LLAMA}

    # Novelty
    echo ">>> $(date "+[%-m-%d-%H:%M:%S]") [Novelty] >>>"
    python -m src.novelty \
    --sequence-file ${EVAL_DIR}/designed.json \
    --evaluation-file ${EVAL_DIR}/novelty.json \
    --mmseqs_path ${MMSEQS_EX} \
    --database_path ${MMSEQS_DB} \
    --num-workers ${NUM_DEVICES}
    # endregion GPU

    # region 2.2.NO GPU
    # Repetitiveness
    echo ">>> $(date "+[%-m-%d-%H:%M:%S]") [Repetitiveness] >>>"
    python -m src.repetitiveness \
    --num-workers ${NUM_WORKERS} \
    --sequence-file ${EVAL_DIR}/designed.json \
    --evaluation-file ${EVAL_DIR}/repetitiveness.json

    # Keyword Recovery
    echo ">>> $(date "+[%-m-%d-%H:%M:%S]") [Keyword Recovery] >>>"
    python -m src.keyword_recovery \
    --num-workers ${NUM_WORKERS_INTERPRO} \
    --sequence-file ${EVAL_DIR}/designed.json \
    --evaluation-file ${EVAL_DIR}/keyword_recovery.json \
    --workers_per_scan ${WORKERS_PER_INTERPRO} \
    --interpro_scan_path ${INTERPRO_SCAN_EX}

    # TMscore
    echo ">>> $(date "+[%-m-%d-%H:%M:%S]") [TMscore] >>>"
    python -m src.tm_score \
    --num-workers ${NUM_WORKERS} \
    --sequence-file ${EVAL_DIR}/designed.json \
    --evaluation-file ${EVAL_DIR}/tm_score.json \
    --output_pdb_dir ${EVAL_DIR}/pdb_esmfold_v1 \
    --tm_score_path ${TM_SCORE_EX}

    # Identity
    echo ">>> $(date "+[%-m-%d-%H:%M:%S]") [Identity] >>>"
    python -m src.identity \
    --num-workers ${NUM_WORKERS} \
    --sequence-file ${EVAL_DIR}/designed.json \
    --evaluation-file ${EVAL_DIR}/identity.json \
    --mmseqs_path ${MMSEQS_EX}
    endregion no-gpu
done

# region Diversity
echo ">>> $(date "+[%-m-%d-%H:%M:%S]") [Diversity] >>>"
python -m src.eval.diversity \
--num-epoches ${#EPOCHES[@]} \
--eval_dir ${EVAL_DIR} \
--sequence-file-name designed.json \
--evaluation-file ${model_dir}/diversity.json \
--mmseqs_path ${MMSEQS_EX} \
--num-workers ${NUM_DEVICES}
# endregion Diversity

# endregion Evaluation
