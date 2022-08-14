#!/bin/bash
export DIR="$(dirname "$(pwd)")"
# conda env update --file ${DIR}'/uncertainty_guided_env.yml'
source activate uncertainty_guided_env
export PYTHONPATH=${PYTHONPATH}:${DIR}
export BASE_PATH='molecular'

export train_path=${BASE_PATH}'/data/zinc/zinc_syn/all_new.txt'
export vocab_path=${BASE_PATH}'/data/zinc/zinc_syn/zinc_vocab_syn.txt'
export prop_path=${BASE_PATH}'/data/zinc/zinc_syn/train_new.lnKD_PK-SA'
export save_path=${BASE_PATH}'/checkpoints/jtvae_drop_MLP0.2_GRU0.2_Prop0.2_zdim56_hidden450_prop_SA_cond_lnKD'

export bs=32
export dropout_rate_MLP=0.2
export dropout_rate_GRU=0.2

export hidden_size=450
export latent_size=56

export property="synthetic_accessibility"
export drop_prop_NN=0.2
export wandb_job_name="pretrain_prop_lnKD"

python JTVAE/fast_molvae/jtnnvae-prop_pretrain.py \
            --train_path ${train_path} \
            --vocab_path  ${vocab_path} \
            --prop_path ${prop_path} \
            --save_path ${save_path} \
            --batch_size ${bs} \
            --hidden_size ${hidden_size} \
            --latent_size ${latent_size} \
            --dropout_rate_MLP ${dropout_rate_MLP} \
            --dropout_rate_GRU ${dropout_rate_GRU} \
            --drop_prop_NN ${drop_prop_NN} \
            --property ${property} \
            --wandb_job_name ${wandb_job_name}

