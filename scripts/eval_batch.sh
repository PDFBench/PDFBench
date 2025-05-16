# Script for Baselines

# region Basement
export TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_EVALUATE_OFFLINE=1
# PDFBench Project Root
PROJECT_ROOT=path/to/PDFBench
# PDFBench Evaluation Root
EVAL_ROOT=${PROJECT_ROOT}/example
# ProTrek weights
PROTREK=path/to/ProTrek-650M/weights/folder
# MMseqs2
MMSEQS_EX=path/to/MMSeqs2/executable
MMSEQS_DB=path/to/mmseqs/DB/uniprotdb
# Interproscan
INTERPRO_SCAN_EX=path/to/interproscan/interproscan-5.74-105.0/interproscan.sh
NUM_WORKERS_INTERPRO=1
WORKERS_PER_INTERPRO=$(( $(nproc) / ${NUM_WORKERS_INTERPRO} ))
# TMscore
TM_SCORE_EX=path/to/Tmscore/executable
# InstPool
MolinstPool=/home/jhkuang/data/cache/dynamsa/data/Molinst/inst2seq.json
ESMWPool=/home/jhkuang/data/cache/dynamsa/data/UniInPro/Inst2seq_w.small.json
ESMWOPool=/home/jhkuang/data/cache/dynamsa/data/UniInPro/Inst2seq_wo.small.json
typeset -A EVAL2POOL=(
    [test_molinst_denovo]=${MolinstPool}
    [test_esm_w_denovo]=${ESMWPool}
    [test_esm_wo_denovo]=${ESMWOPool}
)
# Epoches
EPOCHES=(1 2 3)
# Task
typeset -A EVAL2TASK=(
    [test_molinst_denovo]=molinst
    [test_esm_w_denovo]=esm
    [test_esm_wo_denovo]=esm
)
# endregion Basement

# region Device
# NO-GPU Workers
NUM_WORKERS=$(( $(nproc) / 6 ))

# GPU Workers
export CUDA_VISIBLE_DEVICES=0,1,2,3
NUM_DEVICES=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')

echo ">>> [Running Environment] >>>"
echo NUM_DEVICES: ${NUM_DEVICES}
echo CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}
echo NUM_WORKERS: ${NUM_WORKERS}
echo WORKERS_PER_INTERPRO: ${WORKERS_PER_INTERPRO}
# endregion Device

# region Metrics, Testsets and Models
MOLINST_MODELS=(
    # ours
    # baseline_chroma
    # baseline_paag_annot
    # baseline_paag_annot_ft
    baseline_pinal_small
    # baseline_proteindt_swissprotclap
    # baseline_proteindt_swissprotmolinst
    # baseline_random_empirical
    # baseline_random_uniform
    baseline_random_empirical_plus
)
ESMW_MODELS=(
    # baseline_chroma
    # baseline_esm3_multiword_fixedlen
    # baseline_paag_locat
    # baseline_paag_locat_ft
    # baseline_pinal_small
    # baseline_proteindt_swissinpro
    # baseline_proteindt_swissprotclap 
    # baseline_random_empirical
    baseline_random_uniform
)
ESMWO_MODELS=(
    # ours32
    # ours
    # baseline_chroma
    # baseline_esm3_multiword_fixedlen
    # baseline_paag_locat
    # baseline_paag_locat_ft
    # baseline_pinal_small
    # baseline_proteindt_swissinpro
    # baseline_proteindt_swissprotclap 
    # baseline_random_empirical
    # baseline_random_uniform
    baseline_random_empirical_plus
)
typeset -A METRIC2RUN=(
    # GPU-based
    # ["Novelty"]=1
    # ["Bert-like"]=1
    # ["Perplexity"]=1
    # ["Foldability"]=1
    # ["Language Alignment"]=1
    # ["Retrieval Accuracy"]=1
    # CPU-based
    ["TMscore"]=1
    ["Identity"]=1
    ["Diversity"]=1
    # ["Repetitiveness"]=1
    # ["Keyword Recovery"]=1
)
EVALS=(
    # test_molinst_denovo
    test_esm_wo_denovo
    # test_esm_w_denovo
)
# endregion

typeset -A EVAL2MDOELS=(
    [test_molinst_denovo]=${MOLINST_MODELS}
    [test_esm_w_denovo]=${ESMW_MODELS}
    [test_esm_wo_denovo]=${ESMWO_MODELS}
)

cd ${PROJECT_ROOT}
for EVAL_VAR in ${EVALS}; do
    echo EVAL: ${EVAL_VAR}
    MODELS=(`echo ${EVAL2MDOELS[${EVAL_VAR}]} | tr ' ' ' '`)
    echo MODELS: ${MODELS[@]}
    for MODEL_VAR in ${MODELS}; do
        InstPool=${EVAL2POOL[${EVAL_VAR}]}
        eval_dir=${EVAL_ROOT}/${EVAL_VAR}
        model_dir=${eval_dir}/${MODEL_VAR}

        # region GPU
        for EPOCH in ${EPOCHES[@]}; do
            save_dir=${model_dir}/${EPOCH}

            # # Arguments
            # echo ">>> $(date "+[%-m-%d-%H:%M:%S]") [Arguments] of [${MODEL_VAR}#${EPOCH}] on [${EVAL_VAR}] >>>"
            # echo eval_dir: ${eval_dir}
            # echo model_dir: ${model_dir}
            # echo save_dir: ${save_dir}

            # Path check
            if [ ! -d ${save_dir} ]; then
                echo ${save_dir} does not exist >&2
                continue
            fi

            # Perplexity
            if (( ${METRIC2RUN[Perplexity]} )); then
                echo ">>> $(date "+[%-m-%d-%H:%M:%S]") [Perplexity] of [${MODEL_VAR}#${EPOCH}] on [${EVAL_VAR}] >>>"
                python -m src.eval.perplexity \
                --num-workers $NUM_DEVICES \
                --sequence-file $save_dir/molinst.json \
                --evaluation-file $save_dir/perplexity.json \
                --evaluation_dir $eval_dir \
                --save-plot ${SAVE_PLOT}
            fi

            # Retrieval Accuracy
            if (( ${METRIC2RUN[Retrieval Accuracy]} )); then
                echo ">>> $(date "+[%-m-%d-%H:%M:%S]") [Retrieval Accuracy] of [${MODEL_VAR}#${EPOCH}] on [${EVAL_VAR}] >>>"
                python -m src.eval.retrieval_accuracy \
                --sequence-file ${save_dir}/molinst.json \
                --evaluation-file ${save_dir}/retrieval_accuracy.json \
                --evaluation-dir ${eval_dir} \
                --model_path ${PROTREK} \
                --task ${EVAL2TASK[${EVAL_VAR}]} \
                --num-workers ${NUM_DEVICES} \
                --save-plot ${SAVE_PLOT}
            fi

            # Bert-like
            if (( ${METRIC2RUN[Bert-like]} )); then
                echo ">>> $(date "+[%-m-%d-%H:%M:%S]") [Bert-like] of [${MODEL_VAR}#${EPOCH}] on [${EVAL_VAR}] >>>"
                python -m src.eval.bertlike \
                --num-workers $NUM_DEVICES \
                --sequence-file $save_dir/molinst.json \
                --evaluation-file $save_dir/bertlike.json \
                --evaluation_dir $eval_dir \
                --save-plot ${SAVE_PLOT}
            fi

            # Foldability
            if (( ${METRIC2RUN[Foldability]} )); then
                echo ">>> $(date "+[%-m-%d-%H:%M:%S]") [Foldability] of [${MODEL_VAR}#${EPOCH}] on [${EVAL_VAR}] >>>"
                python -m src.eval.foldability \
                --num-workers $NUM_DEVICES \
                --sequence-file $save_dir/molinst.json \
                --evaluation-file $save_dir/foldability.json \
                --output-pdb-dir $save_dir/pdb_esmfold_v1 \
                --evaluation_dir ${eval_dir} \
                --save-plot ${SAVE_PLOT}
            fi

            # Language Alignment
            if (( ${METRIC2RUN[Language Alignment]} )); then
                echo ">>> $(date "+[%-m-%d-%H:%M:%S]") [Language Alignment] of [${MODEL_VAR}#${EPOCH}] on [${EVAL_VAR}] >>>"
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
            fi

            # Novelty
            if (( ${METRIC2RUN[Novelty]} )); then
                echo ">>> $(date "+[%-m-%d-%H:%M:%S]") [Novelty] of [${MODEL_VAR}#${EPOCH}] on [${EVAL_VAR}] >>>"
                python -m src.eval.novelty \
                --sequence-file ${save_dir}/molinst.json \
                --evaluation-file ${save_dir}/novelty.json \
                --mmseqs_path ${MMSEQS_EX} \
                --database_path ${MMSEQS_DB} \
                --num-workers ${NUM_DEVICES} \
                --evaluation_dir ${eval_dir} \
                --save-plot ${SAVE_PLOT}
            fi
        done
        # endregion GPU

        # region NO-GPU
        for EPOCH in ${EPOCHES[@]}; do
            save_dir=${model_dir}/${EPOCH}

            #   # Arguments
            # echo ">>> [Arguments] of [${MODEL_VAR}#${EPOCH}] on [${EVAL_VAR}] >>>"
            # echo eval_dir: ${eval_dir}
            # echo save_dir: ${save_dir}

            # Repetitiveness
            if (( ${METRIC2RUN[Repetitiveness]} )); then
                echo ">>> $(date "+[%-m-%d-%H:%M:%S]") [Repetitiveness] of [${MODEL_VAR}#${EPOCH}] on [${EVAL_VAR}] >>>"
                python -m src.eval.repetitiveness \
                --num-workers ${NUM_WORKERS} \
                --sequence-file ${save_dir}/molinst.json \
                --evaluation-file ${save_dir}/repetitiveness.json
            fi

            # Keyword Recovery
            if (( ${METRIC2RUN[Keyword Recovery]} )); then
                echo ">>> $(date "+[%-m-%d-%H:%M:%S]") [Keyword Recovery] of [${MODEL_VAR}#${EPOCH}] on [${EVAL_VAR}] >>>"
                python -m src.eval.keyword_recovery \
                --num-workers ${NUM_WORKERS_INTERPRO} \
                --sequence-file ${save_dir}/molinst.json \
                --evaluation-file ${save_dir}/keyword_recovery.json \
                --workers_per_scan ${WORKERS_PER_INTERPRO} \
                --interpro_scan_path ${INTERPRO_SCAN_EX} \
                --evaluation_dir ${eval_dir} \
                --save-plot ${SAVE_PLOT}
            fi

            # TMscore
            if (( ${METRIC2RUN[TMscore]} )); then
                echo ">>> $(date "+[%-m-%d-%H:%M:%S]") [TMscore] of [${MODEL_VAR}#${EPOCH}] on [${EVAL_VAR}] >>>"
                python -m src.eval.tm_score \
                --num-workers ${NUM_WORKERS} \
                --sequence-file ${save_dir}/molinst.json \
                --evaluation-file ${save_dir}/tm_score.json \
                --ref_pdb_dir ${eval_dir}/ground_truth/pdb_esmfold_v1 \
                --res_pdb_dir ${save_dir}/pdb_esmfold_v1 \
                --tm_score_path ${TM_SCORE_EX} \
                --evaluation_dir ${eval_dir} \
                --save-plot ${SAVE_PLOT}
            fi

            # Identity
            if (( ${METRIC2RUN[Identity]} )); then
                echo ">>> $(date "+[%-m-%d-%H:%M:%S]") [Identity] of [${MODEL_VAR}#${EPOCH}] on [${EVAL_VAR}] >>>"
                python -m src.eval.identity \
                --num-workers ${NUM_WORKERS} \
                --sequence-file ${save_dir}/molinst.json \
                --evaluation-file ${save_dir}/identity.json \
                --mmseqs_path ${MMSEQS_EX} \
                --evaluation_dir ${eval_dir} \
                --save-plot ${SAVE_PLOT}
            fi
        done
        # endregion no-gpu

        # Diversity
        if (( ${METRIC2RUN[Diversity]} == 1 )); then
            echo ">>> $(date "+[%-m-%d-%H:%M:%S]") [Diversity] of [${MODEL_VAR}#${EPOCH}] on [${EVAL_VAR}] >>>"
            python -m src.eval.diversity \
            --num-epoches ${#EPOCHES[@]} \
            --model-dir ${model_dir} \
            --evaluation-file ${model_dir}/diversity.json \
            --mmseqs_path ${MMSEQS_EX} \
            --num-workers ${NUM_DEVICES} \
            --evaluation_dir ${eval_dir} \
            --save-plot ${SAVE_PLOT}
        fi
        # endregion Diversity
    done

    # Display
    python -m src.eval._metric \
    --evaluation_dir ${eval_dir}
done