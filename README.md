# CONDITIONAL JTVAE

Official repository for the paper (research): Conditional-JT-VAE for designing molecules with multiple specific properties.  
This repo is forked from https://github.com/pascalnotin/uncertainty_guided_optimization.

## Junction-Tree VAE (JTVAE)
We extend the molecular optimization approach described in [Junction Tree Variational Autoencoder for Molecular Graph Generation](https://arxiv.org/abs/1802.04364) by Jin et al., and build on top of the corresponding codebase: https://github.com/wengong-jin/icml18-jtnn.

This repository includes the following enhancements:
- Extending the JTNNVAE class with methods to quantify decoder uncertainty in latent.
- Creating a separate subclass (JTNNVAE_prop) to facilitate the joint training of a JTVAE with an auxiliary network predicting a property of interest (eg., penalized logP).
- Including additional hyperparameters to apply dropout in the JTVAE architecture (thereby supporting weight sampling via MC dropout).
- Providing new functionalities to perform uncertainty-guided optimization in latent and assess the quality of generated molecules.
- Supporting a more comprehensive set of Bayesian Optimization functionalities via [BoTorch](https://botorch.org/).
- Migrating the original codebase to a more recent software stack (Python v3.8 and Pytorch v1.10.0).

Example scripts are provided in `scripts/` to:
1. Preprocess the data (JTVAE_data_preprocess.sh) and generate a new vocabulary on new dataset (JTVAE_data_vocab_generation.sh)
2. Train the JTVAE networks:
- JTVAE with no auxiliary property network: JTVAE_train_jtnnvae.sh
- JTVAE with auxiliary property network: 
    - JTVAE_train_jtnnvae-prop_step1_pretrain.sh to pre-train the joint architecture (with no KL term in the loss)
    - JTVAE_train_jtnnvae-prop_step2_train.sh to train the joint architecture
3. Test the quality of trained JTVAE networks (JTVAE_test_jtnnvae.sh and JTVAE_test_jtnnvae-prop.sh)
4. Perform uncertainty-guided optimization in latent:
- Gradient ascent: JTVAE_uncertainty_guided_optimization_gradient_ascent.sh
- Bayesian optimization: JTVAE_uncertainty_guided_optimization_bayesian_optimization.sh

## Environment setup
The required environment may be created via conda and the provided uncertainty_guided_env.yml file as follows:
```
  conda env create -f uncertainty_guided_env.yml
  conda activate uncertainty_guided_env
```
