export DIR="$(dirname "$(pwd)")"
# conda env update --file ${DIR}'/uncertainty_guided_env.yml'
source activate uncertainty_guided_env
export PYTHONPATH=${PYTHONPATH}:${DIR}
export BASE_PATH='molecular'

export nsample=10000
export vocab=${BASE_PATH}'/data/zinc/zinc_syn/zinc_vocab_syn.txt'
export
model_checkpoint=${BASE_PATH}'/checkpoints/jtvae_drop_MLP0.2_GRU0.2_Prop0.2_zdim56_hidden450_prop_SA_cond_lnKD/model.pre_trained_epoch-1'
export output_file='20220809'

export hidden_size=450
export latent_size=56
export depthT=20
export depthG=3
export dropout_rate_GRU=0.2
export dropout_rate_MLP=0.2
export property="synthetic_accessibility"
export drop_prop_NN=0.2
export cond_lnKD_size=14
export cond_lnKD=-12

python JTVAE/fast_molvae/sample.py \
            --nsample ${nsample} \
            --vocab  ${vocab} \
            --model_checkpoint ${model_checkpoint} \
            --output_file ${output_file} \
            --hidden_size ${hidden_size} \
            --latent_size ${latent_size} \
            --depthT ${depthT} \
            --depthG ${depthG} \
            --dropout_rate_GRU ${dropout_rate_GRU} \
            --dropout_rate_MLP ${dropout_rate_MLP} \
            --property ${property} \
            --drop_prop_NN ${drop_prop_NN} \
            --cond_lnKD_size ${cond_lnKD_size} \
            --cond_lnKD ${cond_lnKD}
