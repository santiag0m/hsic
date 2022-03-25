# robust-hsic

This repository aims to replicate the results from the ICML2020 paper "Robust Learning with the Hilbert-Schmidt Independence Criterion"

- Paper: http://proceedings.mlr.press/v119/greenfeld20a/greenfeld20a-supp.pdf
- Original Implementation: https://github.com/danielgreenfeld3/XIC

## Preliminary results

### Synthetic Data

Training Params:

- Batch Size: 32
- Learning Rate: 1e-3
- L2 Regularization (Weight Decay): 1e-3
- Num Epochs (Best of): 10

Experiment Params:

- Noise Distribution: Gaussian
- Loss Function: HSIC
- Num. Trials: 20

![Dataset Size vs. MSE graph](https://github.com/santiag0m/robust-hsic/blob/main/results/gaussian_hsic_bs32_lr1e3_wd1e3.png)
