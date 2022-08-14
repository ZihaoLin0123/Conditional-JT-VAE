export DIR="$(dirname "$(pwd)")"
# conda env update --file ${DIR}'/uncertainty_guided_env.yml'
# source activate uncertainty_guided_env
export PYTHONPATH=${PYTHONPATH}:${DIR}

export nsample=20  # 100
export vocab_path=  # please specify your own vocab_path
export model_checkpoint=  # please specify your own model_checkpoint
export output_file='20210101'

export hidden_size=450
export latent_size=56
export depthT=20
export depthG=3
export dropout_rate_GRU=0.2
export dropout_rate_MLP=0.2

python JTVAE/fast_molvae/sample.py \
            --nsample ${nsample} \
            --vocab_path  ${vocab_path} \
            --model_checkpoint ${model_checkpoint} \
            --output_file ${output_file} \
            --hidden_size ${hidden_size} \
            --latent_size ${latent_size} \
            --depthT ${depthT} \
            --depthG ${depthG} \
            --dropout_rate_GRU ${dropout_rate_GRU} \
            --dropout_rate_MLP ${dropout_rate_MLP}
            
