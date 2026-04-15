# export TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_EVALUATE_OFFLINE=1
date

test_dir=PATH/TO/TEST_DIR   # TODO: specify the path to the test directory
mkdir -p ${test_dir}

# TODO: specify the path to the test dataset
python func2seq.py \
--input_path PATH/TO/SWISSTEST \
--output_path ${test_dir}/designed.json \
--random_length False \
--num_keyword None \
--prefix 0 