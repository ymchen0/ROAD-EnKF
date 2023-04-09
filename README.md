# Reduced-Order Autodifferentiable Ensemble Kalman Filters (ROAD-EnKF)

Joint learning of latent reduced-order dynamics and states from noisy observations, by auto-differentiating through an Ensemble Kalman Filter (EnKF) using PyTorch. This repo is built upon AD-EnKF (https://github.com/ymchen0/torchEnKF). 

Compared to AD-EnKF, we enable nonlinear observation model with spectral convolutional layers (see Sec 4.1 of paper). Similar parameterization can also be used for the dynamics model in AD-EnKF to increase flexibility (see Appendix A). The implementation can be found in `torchEnKF/nn_templates.py`.

(More examples coming soon...)

