import torch
from torch.distributions import LogNormal, Normal, Independent
import numpy as np
from scipy.optimize import minimize, Bounds
from time import time

torch.manual_seed(0)

def get_x(theta, design):
  theta1 = theta[:, 0:1]
  theta2 = theta[:, 1:2]
  theta3 = theta[:, 2:3]
  x = 400. * theta2 * (torch.exp(-theta1*design) - torch.exp(-theta2*design)) / (theta3*(theta2-theta1))
  return x

n_inner = 1000
n_outer = 100
loc = torch.tensor(np.log((0.1, 1., 20.)), dtype=torch.float64)
scale = torch.tensor(np.sqrt((0.05, 0.05, 0.05)), dtype=torch.float64)
prior = LogNormal(loc, scale)
prior = Independent(prior, 1)
theta_inner = prior.sample((n_inner,))
theta_outer = prior.sample((n_outer,))
loc = torch.zeros(15, dtype=torch.float64)
scale = 0.1 * torch.ones(15, dtype=torch.float64)
noise = Normal(loc, scale)
noise = Independent(noise, 1)
noise_entropy = noise.entropy()
noise_outer = noise.sample((n_outer,))

def objective(design):
  x_outer = get_x(theta_outer, design)
  x_inner = get_x(theta_inner, design)
  y_outer = x_outer + noise_outer
  # Get matrix of all y_outer-x_inner values
  diff = y_outer.unsqueeze(1) - x_inner.unsqueeze(0)
  log_prob_diff = noise.log_prob(diff)
  log_evidence = torch.logsumexp(log_prob_diff, dim=1) - np.log(n_inner)
  sig = noise_entropy - log_evidence.mean()
  print('Design ', np.sort(design))
  print('SIG {:.3f}'.format(sig.numpy()))
  return -sig.numpy()


np.set_printoptions(precision=3, suppress=True)

design0 = np.random.uniform(0.,24.,15)
bounds = Bounds(lb = np.zeros(15), ub = 24.*np.ones(15))
start_time = time()
out = minimize(objective, design0, method='L-BFGS-B', bounds=bounds)
print('Time taken: {:1f} seconds'.format(time() - start_time))



