#!/bin/bash

export DIR="$(dirname "$(pwd)")"
export DATADIR="/usr/xtmp/yy331/molecular"
# conda env update --file ${DIR}'/uncertainty_guided_env.yml'
source activate uncertainty_guided_env
export PYTHONPATH=${PYTHONPATH}:${DIR}

export processed_data_path=${DATADIR}'/data/zinc_processed_final_aug'
export save_path=${DATADIR}'/checkpoints/jtvae_drop_MLP0.2_GRU0.2_zdim56_hidden450_cond_lnKD_SelectPtoM_final_20220805/'
export vocab_path=${DATADIR}'/data/zinc/zinc_final_aug/zinc_vocab_final_aug.txt'  # new_vocab.txt
export cond_lnKD_path=${DATADIR}'/data/zinc/zinc_final_aug/train_new.lnKD_PK-SA'
export cond_SelectPtoM_path=${DATADIR}'/data/zinc/zinc_final_aug/train_new.Select_PtoM-SA'

export bs=32
export dropout_rate_MLP=0.2
export dropout_rate_GRU=0.2
export hidden_size=450
export latent_size=56
export cond_lnKD_size=14
export cond_SelectPtoM_size=18
export wandb_job_name='jtvae_lnKD_SelectPtoM_final_aug_20220805'

python JTVAE/fast_molvae/jtnnvae_train.py \
            --train_path ${processed_data_path} \
            --vocab_path  ${vocab_path} \
            --cond_lnKD_path ${cond_lnKD_path} \
            --cond_lnKD_size  ${cond_lnKD_size} \
            --cond_SelectPtoM_size  ${cond_SelectPtoM_size} \
            --cond_SelectPtoM_path  ${cond_SelectPtoM_path} \
            --save_path ${save_path} \
            --batch_size ${bs} \
            --hidden_size ${hidden_size} \
            --latent_size ${latent_size} \
            --dropout_rate_MLP ${dropout_rate_MLP} \
            --dropout_rate_GRU ${dropout_rate_GRU} \
            --wandb_job_name ${wandb_job_name}
