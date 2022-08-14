#!/bin/bash
# export DIR="$(dirname "$(pwd)")"
# conda env update --file ${DIR}'/uncertainty_guided_env.yml'

source activate uncertainty_guided_env
export PYTHONPATH=${PYTHONPATH}:${DIR}

python JTVAE/fast_molvae/data_preprocess.py \
                --train  'train.txt' \  # please specify your own train data file path
                --output 'output_folder_path'  # please specify your own output folder path
                # --split 1 --jobs 1 \
