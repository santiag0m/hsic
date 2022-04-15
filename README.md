# robust-hsic

This repository aims to replicate the results from the ICML2020 paper "Robust Learning with the Hilbert-Schmidt Independence Criterion"

- Paper: http://proceedings.mlr.press/v119/greenfeld20a/greenfeld20a-supp.pdf
- Original Implementation: https://github.com/danielgreenfeld3/XIC

## Results

### Synthetic Data

Training Params:

- Batch Size: 32
- Learning Rate: 1e-3
- L2 Regularization (Weight Decay): 1e-3
- Num Epochs (Best of): 10

Experiment Params:

- Loss Function: HSIC
- Num. Trials: 20

![Dataset Size vs. MSE graph with Gaussian Noise](https://github.com/santiag0m/robust-hsic/blob/main/results/gaussian_hsic_bs32_lr1e3_wd1e3.png)

![Dataset Size vs. MSE graph with Shifted Exponential Noise](https://github.com/santiag0m/robust-hsic/blob/main/results/shifted_exponential_hsic_bs32_lr1e3_wd1e3.png)

![Dataset Size vs. MSE graph with Laplacian Noise](https://github.com/santiag0m/robust-hsic/blob/main/results/laplacian_hsic_bs32_lr1e3_wd1e3.png)

### Rotated MNIST

Training Params:

- Batch Size: 32
- Learning Rate: 1e-3
- L2 Regularization (Weight Decay): None
- Num Epochs (Best of): 7

![HSIC and Cross Entropy vs. Accuracy](https://github.com/santiag0m/hsic/blob/main/results/rotated_mnist_grid.png)
