# Scripts for evaluation of PDFBench
# This script evaluate single designed result.

# region 1.Settings(See README.md, same as eval.sh)
export TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_EVALUATE_OFFLINE=1
# PDFBench Project Root
PROJECT_ROOT=/path/to/PDFBench
# PDFBench Evaluation Root
EVAL_DIR=${PROJECT_ROOT}/example/batch

# region 1.1.Evaluation Settings
# ProTrek weights
PROTREK=/path/to/ProTrek-650M/weights/folder
# MMseqs2
MMSEQS_EX=/path/to/mmseqs/bin/mmseqs
MMSEQS_DB=/path/to/mmseqs/DB/uniprotdb_gpu
# Interproscan
INTERPRO_SCAN_EX=/path/to/interproscan/interproscan-5.73-104.0/interproscan.sh
# TMscore
TM_SCORE_EX=/path/to/TMscore/TMscore
# Diversity

# endregion Basement

# region 1.2.Device Settings
NUM_WORKERS=$(( $(nproc) / 6 )) # NO-GPU Workers
export CUDA_VISIBLE_DEVICES=0,1,2,3 # GPU Workers
WORKERS_PER_INTERPRO=$(( $(nproc) / ${NUM_WORKERS_INTERPRO} ))  # Workers for InterProScan
NUM_DEVICES=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
# endregion Device

# region 1.3.Batch Settings
EPOCHES=(1 2 3) # We evaluate all baseline for 3 epoches.

# endregion

# region 1.4.Setting Display
echo ">>> [Running Environment] >>>"
echo NUM_DEVICES: ${NUM_DEVICES}
echo CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}
echo NUM_WORKERS: ${NUM_WORKERS}
echo WORKERS_PER_INTERPRO: ${WORKERS_PER_INTERPRO}
echo "<<< [Running Environment] <<<"
# endregion

# endregion 1.Settings

# region 2.Evaluation
cd ${PROJECT_ROOT}
for epoch in ${EPOCHES[@]}; do
    if [ ! -d ${save_dir} ]; then   # Path check
        echo ${save_dir} does not exist >&2
        exit 1
    fi
    eval_dir=${model_dir}/${epoch}
    # region 2.1.GPU-based
    # Perplexity
    echo ">>> $(date "+[%-m-%d-%H:%M:%S]") [Perplexity#${EPOCH}]>>>"
    python -m src.eval.perplexity \
    --num-workers $NUM_DEVICES \
    --sequence-file $save_dir/molinst.json \
    --evaluation-file $save_dir/perplexity.json \
    --evaluation_dir $eval_dir \
    --save-plot ${SAVE_PLOT}

    # Retrieval Accuracy
    echo ">>> $(date "+[%-m-%d-%H:%M:%S]") [Retrieval Accuracy#${EPOCH}] >>>"
    python -m src.eval.retrieval_accuracy \
    --sequence-file ${save_dir}/molinst.json \
    --evaluation-file ${save_dir}/retrieval_accuracy.json \
    --evaluation-dir ${eval_dir} \
    --model_path ${PROTREK} \
    --task ${EVAL2TASK[${EVAL_VAR}]} \
    --num-workers ${NUM_DEVICES} \
    --save-plot ${SAVE_PLOT}

    # Bert-like
    echo ">>> $(date "+[%-m-%d-%H:%M:%S]") [Bert-like#${EPOCH}] >>>"
    python -m src.eval.bertlike \
    --num-workers $NUM_DEVICES \
    --sequence-file $save_dir/molinst.json \
    --evaluation-file $save_dir/bertlike.json \
    --evaluation_dir $eval_dir \
    --save-plot ${SAVE_PLOT}

    # Foldability
    echo ">>> $(date "+[%-m-%d-%H:%M:%S]") [Foldability#${EPOCH}] >>>"
    python -m src.eval.foldability \
    --num-workers $NUM_DEVICES \
    --sequence-file $save_dir/molinst.json \
    --evaluation-file $save_dir/foldability.json \
    --output-pdb-dir $save_dir/pdb_esmfold_v1 \
    --evaluation_dir ${eval_dir} \
    --save-plot ${SAVE_PLOT}

    # Language Alignment
    echo ">>> $(date "+[%-m-%d-%H:%M:%S]") [Language Alignment#${EPOCH}] >>>"
    python -m src.eval.language_alignment \
    --num-workers $NUM_DEVICES \
    --sequence-file $save_dir/molinst.json \
    --evaluation-file $save_dir/language_alignment.json \
    --task ${EVAL2TASK[${EVAL_VAR}]} \
    --use-structure False \
    --use-sequence True \
    --pdb-dir $save_dir/pdb_esmfold_v1 \
    --protrek-path ${PROTREK} \
    --evollama-path /home/nwliu/data/pretrain/EvoLlama/oracle_denovo_3B/checkpoint-50000 \
    --llm-path /home/nwliu/data/pretrain/Llama-3.2-3B-Instruct \
    --evaluation_dir ${eval_dir} \
    --save-plot ${SAVE_PLOT}

    # Novelty
    echo ">>> $(date "+[%-m-%d-%H:%M:%S]") [Novelty#${EPOCH}] >>>"
    python -m src.eval.novelty \
    --sequence-file ${save_dir}/molinst.json \
    --evaluation-file ${save_dir}/novelty.json \
    --mmseqs_path ${MMSEQS_EX} \
    --database_path ${MMSEQS_DB} \
    --num-workers ${NUM_DEVICES} \
    --evaluation_dir ${eval_dir} \
    --save-plot ${SAVE_PLOT}
    # endregion GPU

    # region NO-GPU
    # Repetitiveness
    echo ">>> $(date "+[%-m-%d-%H:%M:%S]") [Repetitiveness#${EPOCH}] >>>"
    python -m src.eval.repetitiveness \
    --num-workers ${NUM_WORKERS} \
    --sequence-file ${save_dir}/molinst.json \
    --evaluation-file ${save_dir}/repetitiveness.json

    # Keyword Recovery
    echo ">>> $(date "+[%-m-%d-%H:%M:%S]") [Keyword Recovery#${EPOCH}] >>>"
    python -m src.eval.keyword_recovery \
    --num-workers ${NUM_WORKERS_INTERPRO} \
    --sequence-file ${save_dir}/molinst.json \
    --evaluation-file ${save_dir}/keyword_recovery.json \
    --workers_per_scan ${WORKERS_PER_INTERPRO} \
    --interpro_scan_path ${INTERPRO_SCAN_EX} \
    --evaluation_dir ${eval_dir} \
    --save-plot ${SAVE_PLOT}

    # TMscore
    echo ">>> $(date "+[%-m-%d-%H:%M:%S]") [TMscore#${EPOCH}] >>>"
    python -m src.eval.tm_score \
    --num-workers ${NUM_WORKERS} \
    --sequence-file ${save_dir}/molinst.json \
    --evaluation-file ${save_dir}/tm_score.json \
    --ref_pdb_dir ${eval_dir}/ground_truth/pdb_esmfold_v1 \
    --res_pdb_dir ${save_dir}/pdb_esmfold_v1 \
    --tm_score_path ${TM_SCORE_EX} \
    --evaluation_dir ${eval_dir} \
    --save-plot ${SAVE_PLOT}

    # Identity
    echo ">>> $(date "+[%-m-%d-%H:%M:%S]") [Identity#${EPOCH}] >>>"
    python -m src.eval.identity \
    --num-workers ${NUM_WORKERS} \
    --sequence-file ${save_dir}/molinst.json \
    --evaluation-file ${save_dir}/identity.json \
    --mmseqs_path ${MMSEQS_EX} \
    --evaluation_dir ${eval_dir} \
    --save-plot ${SAVE_PLOT}
    # endregion no-gpu
done

# region Diversity
echo ">>> $(date "+[%-m-%d-%H:%M:%S]") [Diversity] >>>"
python -m src.eval.diversity \
--num-epoches ${#EPOCHES[@]} \
--model-dir ${model_dir} \
--evaluation-file ${model_dir}/diversity.json \
--mmseqs_path ${MMSEQS_EX} \
--num-workers ${NUM_DEVICES} \
--evaluation_dir ${eval_dir} \
--save-plot ${SAVE_PLOT}
# endregion Diversity

# endregion Evaluation
