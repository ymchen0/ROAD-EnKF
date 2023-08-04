import torch
import torch.nn as nn
import math

def compl_mul1d(input, weights):
  # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
  return torch.einsum("bix,iox->box", input, weights)

class Linear(nn.Module):
  def __init__(self, x_dim, y_dim, H):
    super().__init__()
    self.H = nn.Parameter(H)
    self.x_dim = x_dim
    self.y_dim = y_dim

  def forward(self, u):
    # du/dt = f(u, t), input: N * x_dim, output: N * x_dim
    out = u @ self.H.t()
    return out

class Lorenz63(nn.Module):
  def __init__(self, coeff):
    super().__init__()
    self.coeff = nn.Parameter(coeff)

  def forward(self, t, u):
    # (*bs * x_dim) -> (*bs * x_dim)
    sigma, beta, rho = self.coeff
    out = torch.stack((sigma * (u[...,1] - u[...,0]), rho * u[...,0] - u[...,1] - u[...,0] * u[...,2], u[...,0] * u[...,1] - beta * u[...,2]), dim=-1)
    return out

class SpectralConv1d(nn.Module):
  # 1D Fourier layer. It does FFT, linear transform, and Inverse FFT.
  # Borrowed from https://github.com/neuraloperator/neuraloperator
  def __init__(self, in_channels, out_channels, modes1):
    super(SpectralConv1d, self).__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(x_dim/2) + 1

    self.scale = (1 / (in_channels * out_channels))
    self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))

  def forward(self, x):
    batchsize = x.shape[0]
    # Compute Fourier coeffcients up to factor of e^(- something constant)
    x_ft = torch.fft.rfft(x)

    # Multiply relevant Fourier modes
    out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1) // 2 + 1, device=x.device, dtype=torch.cfloat)
    out_ft[:, :, :self.modes1] = compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)

    # Return to physical space
    x = torch.fft.irfft(out_ft, n=x.size(-1))
    return x


class FDec(nn.Module):
  # Fourier Neural Decoder: Decoder with spectral convolutional layers
  def __init__(self, z_dim, x_dim, hidden, width, n_layers, modes=12, ln=True):
    # hidden: size of the hidden layer, h
    # width: channel numbers, = n_1 = n_2 = ... = n_L  (see paper)
    # n_layers: number of layers, L
    # modes: see annotation in 'SpectralConv1d' above
    # ln: do layer normalization if True
    super(FDec, self).__init__()
    self.z_dim = z_dim
    self.x_dim = x_dim
    self.modes = min(modes, x_dim // 2 + 1) # modes should be at most floor(x_dim/2) + 1
    self.width = width
    self.hidden = hidden
    self.n_layers = n_layers
    self.ln = ln

    self.weight0 = nn.Parameter(1 / (z_dim * hidden) * torch.rand(1, z_dim, hidden, dtype=torch.cfloat))
    self.convs = nn.ModuleList()
    self.ws = nn.ModuleList()
    if ln:
      self.norms = nn.ModuleList()
    for i in range(n_layers):
      width_in = 1 if i == 0 else width
      conv_layer = SpectralConv1d(width_in, self.width, self.modes)
      oneone_conv_layer = nn.Conv1d(width_in, self.width, 1, bias=True)
      self.convs.append(conv_layer)
      self.ws.append(oneone_conv_layer)
      if ln:
        norm_layer = nn.LayerNorm(self.x_dim)
        self.norms.append(norm_layer)
    self.fc1 = nn.Linear(self.width, 128)
    self.fc2 = nn.Linear(128, 1)
    self.activation = nn.ReLU()

  def forward(self, z):
    # z: tensor of shape (*bs, z_dim)
    bs = z.shape[:-1]
    z = z.reshape(-1, 1, self.z_dim)  # (bs, 1, z_dim)
    zc = torch.matmul(z.type(torch.cfloat), self.weight0)
    zc = zc.view(-1, 1, self.hidden)  # (bs, 1, hidden)
    x = torch.fft.irfft(zc, n=self.x_dim)  # (bs, 1, x_dim)

    for i in range(self.n_layers):
      x1 = self.convs[i](x)  # (bs, width, x_dim)
      x2 = self.ws[i](x)
      x = x1 + x2
      if self.ln:
        x = self.norms[i](x)
      x = self.activation(x)  # (bs, width, x_dim)

    x = x.permute(0, 2, 1)  # (bs, x_dim, width)
    x = self.fc1(x)  # (bs, x_dim, 128)
    x = self.activation(x)
    x = self.fc2(x)  # (bs, x_dim, 1)

    return x.view(*bs, self.x_dim)


class FODE_Net(nn.Module):
  # ODE Dynamics model (represented by f_\alpha in paper) with spectral convolutional layers
  # Set adjoint=False if you are using this as the 'ode_func' in da_methods.EnKF
  def __init__(self, x_dim, hidden, width, n_layers, modes=12, ln=True):
    # hidden: size of the hidden layer, h
    # width: channel numbers, = n_1 = n_2 = ... = n_L  (see paper)
    # n_layers: number of layers, L
    # modes: see annotation in 'SpectralConv1d' above
    # ln: do layer normalization if True
    super(FODE_Net, self).__init__()
    self.x_dim = x_dim
    self.modes = min(modes, x_dim // 2 + 1) # modes should be at most floor(x_dim/2) + 1
    self.width = width
    self.hidden = hidden
    self.n_layers = n_layers
    self.ln = ln

    self.convs = nn.ModuleList()
    self.ws = nn.ModuleList()
    if ln:
      self.norms = nn.ModuleList()
    for i in range(n_layers):
      width_in = 1 if i == 0 else width
      conv_layer = SpectralConv1d(width_in, self.width, self.modes)
      oneone_conv_layer = nn.Conv1d(width_in, self.width, 1, bias=True)
      self.convs.append(conv_layer)
      self.ws.append(oneone_conv_layer)
      if ln:
        norm_layer = nn.LayerNorm(self.x_dim)
        self.norms.append(norm_layer)
    self.fc1 = nn.Linear(self.width, 128)
    self.fc2 = nn.Linear(128, 1)
    self.activation = nn.GELU()

  def forward(self, t, x):
    # x: tensor of shape (*bs, x_dim)
    bs = x.shape[:-1]
    x = x.reshape(-1, 1, self.x_dim)  # (bs, 1, x_dim)

    for i in range(self.n_layers):
      x1 = self.convs[i](x)  # (bs, width, x_dim)
      x2 = self.ws[i](x)
      x = x1 + x2
      if self.ln:
        x = self.norms[i](x)
      x = self.activation(x)  # (bs, width, x_dim)

    x = x.permute(0, 2, 1)  # (bs, x_dim, width)
    x = self.fc1(x)  # (bs, x_dim, 128)
    x = self.activation(x)
    x = self.fc2(x)  # (bs, x_dim, 1)

    return x.view(*bs, self.x_dim)
