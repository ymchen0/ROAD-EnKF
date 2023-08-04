from tqdm import tqdm
import os
import sys
sys.path.append(os.getcwd()) # Fix Python path

from torchEnKF import da_methods, nn_templates, noise
from examples import generate_data, utils

import random
import torch
import numpy as np
from torchdiffeq import odeint
import matplotlib.pyplot as plt


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device="cpu"
print(f"device: {device}")


seed = 40
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


######### Define reference model #########
z_dim = 3  # Do not change
l63_coeff = torch.tensor([10., 8/3, 28.])
true_latent_ode_func = nn_templates.Lorenz63(l63_coeff).to(device)


######### Warmup: Draw an initial point z0 from the L63 limit cycle. Can be ignored for problems with a smaller scale. #########
with torch.no_grad():
    z0_warmup = torch.distributions.MultivariateNormal(torch.zeros(z_dim), covariance_matrix=100 * torch.eye(z_dim)).sample().to(device)
    t_warmup = torch.tensor([0., 60.]).to(device)
    z0 = odeint(true_latent_ode_func, z0_warmup, t_warmup, method='rk4', options=dict(step_size=0.05))[1]  # Shape (z_dim,)

######### Generate data from the reference model #########
t0 = 0.
t_obs_step = 0.01
n_obs = 300
t_obs = t_obs_step * torch.arange(1, n_obs+1).to(device)
x_dim = 128


def true_decoder(z):
    return torch.cat((z/40, (z/40)**3), dim=-1) @ utils.legendre(6, x_dim).to(device)


indices = [i for i in range(x_dim)]
y_dim = len(indices)
model_S_true = None  # No noise in dynamics
H_true = torch.eye(x_dim)
true_obs_func = nn_templates.Linear(x_dim, y_dim, H=H_true).to(device)  # Full observation
noise_R_true = noise.AddGaussian(y_dim, torch.tensor(0.1), param_type='scalar').to(device)  # Gaussian observation noise with std 0.1
with torch.no_grad():
    z_truth, x_truth, y_obs = generate_data.generate_from_latent(true_latent_ode_func, true_decoder, true_obs_func, t_obs, z0, model_S_true, noise_R_true, device=device,
                                            ode_method='rk4', ode_options=dict(step_size=0.05), tqdm=tqdm)



########## Run EnKF with the reference model, compute log-likelihood and gradient #########
N_ensem = 50
init_m = torch.zeros(z_dim, device=device)
init_C_param = noise.AddGaussian(z_dim, 50 * torch.eye(z_dim), 'full').to(device)
obs_func = lambda z: true_obs_func(true_decoder(z))
Z, res, log_likelihood = da_methods.EnKF(true_latent_ode_func,obs_func, t_obs, y_obs, N_ensem, init_m, init_C_param, model_S_true, noise_R_true,device,dec=true_decoder,
                                              save_filter_step={'mean','decoder_mean'}, tqdm=tqdm)
print(f"log-likelihood estimate: {log_likelihood}")
burn_in = n_obs // 5
print(f"Filter accuracy, latent space (RMSE): {torch.sqrt(utils.mse_loss(res['mean'][burn_in:], z_truth[burn_in:]))}")
print(f"Filter accuracy, state space (RMSE): {torch.sqrt(utils.mse_loss(res['decoder_mean'][burn_in:], x_truth[burn_in:]))}")
print("Computing gradient...")
log_likelihood.backward()
print(f"Gradient: {true_latent_ode_func.coeff.grad}")

### The log-likelihood estimates and gradient estimates can be used for various purposes!
