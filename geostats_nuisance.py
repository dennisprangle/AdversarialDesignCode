import torch
from torch.distributions import MultivariateNormal, Gamma
from torch.autograd.functional import hessian
import numpy as np

ell_prior = Gamma(2., 40.)
## shape-rate parameterisation aka alpha-beta
## Gives expectation = 0.05, sd = 0.035 (approx)
## So similar range to figure in paper
## And density 0 at ell=0, avoiding v small values

sigma_prior = Gamma(2., 2.)
## Gives expectation = 1, sd = 0.71 (approx)
## So similar range to figure in paper
## And density 0 at ell=0, avoiding v small values

def get_var_matrix(phi, squared_diffs):
  """ Calculate variance matrix of geostatitical model

  Inputs:
  `phi` is a tensor of shape (3) (the nuisance parameters: sigma1, sigma2, l)
  `squared_diffs` is a tensor of distances between design locations
  (This has dimension (ndesign, ndesign, 2) and
  element (i,j,k) is dist between kth dimension of points i and j)

  Returns a variance matrix, with shape (ndesign, ndesign)
  """
  sigma1, sigma2, ell = phi
  r_matrix = torch.exp(-torch.sum(squared_diffs/(ell ** 2.), -1))
  return sigma1 ** 2. * r_matrix + sigma2 ** 2. * torch.eye(ndesign)

def sample_phi(theta, phi0, y, design, nsteps=10):
  """ Sample the nuisance parameters using MCMC

  Inputs:
  `theta` is a tensor of shape (2) (parameters of interest)
  `design` is a tensor of shape (ndesign, 2) (observation locations)
  `y` is a tensor of shape (ndesign) (observations)
  `phi0` is a tensor of shape (3) (true nuisance parameters: sigma1, sigma2, l)

  Returns an approx sample from the phi posterior given theta, design, y
  """
  ## (IT MIGHT EVENTUALLY BE POSSIBLE TO BATCH THIS WRT DESIGNS
  ## BUT I'M NOT GOING TO ATTEMPT THAT FOR THIS PAPER.)

  ##Initial calculations
  x = theta[0]*design[:,0] + theta[1]*design[:,1]
  eps = y - x
  squared_diffs = (design.unsqueeze(1) - design.unsqueeze(0)) ** 2.
  ## n.b. squared_diffs matrix calculation could be more efficient
  ## See https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065

  def log_target(phi):
    if torch.any(phi < 0.):
      return torch.tensor(-np.inf)
    sigma1, sigma2, ell = phi
    log_prior = ell_prior.log_prob(ell) + sigma_prior.log_prob(sigma1) \
                + sigma_prior.log_prob(sigma2)
    ## TO DO: quicker to use density formua directly for next line?
    obs_dist = MultivariateNormal(loc=torch.zeros_like(eps),
                                  covariance_matrix=get_var_matrix(phi,
                                                                   squared_diffs))
    return log_prior + obs_dist.log_prob(eps)

  hess = -hessian(log_target, phi0)
  ## PROBLEM: hess not guaranteed to be pos def. So need to optimise log_target?
  ## Less of an issue when ndesign large - target more Gaussian?
  ## For now I'm going to ignore this issue and hope for the best!
  proposal_prec = hess * 3. / (2.38 ** 2.)

  ##Initial MCMC state
  increment_dist = MultivariateNormal(loc=torch.zeros_like(phi0),
                                      precision_matrix=proposal_prec)
  while True:
    phi = phi0 + increment_dist.sample()
    if all(phi > 0.):
      break
  log_target_at_phi = log_target(phi)

  ##MCMC loop
  for i in range(nsteps):
    print("Step", i, "state", phi)
    proposal = phi + increment_dist.sample()
    log_target_at_proposal = log_target(proposal)
    log_ratio = log_target_at_proposal - log_target_at_phi
    print("alpha {:.2}".format(torch.exp(log_ratio)))
    u = np.random.uniform()
    if np.log(u) < log_ratio:
      phi = proposal
      log_target_at_phi = log_target_at_proposal

  print("Step", nsteps, "state", phi)
  return phi


##Create some inputs
np.random.seed(1)
theta = torch.tensor((-2.,1.0), dtype=torch.float32)
ndesign = 500 ## Same value as in paper
design = np.random.uniform(-0.5, 0.5, (ndesign,2))
design = torch.tensor(design, dtype=torch.float32)
squared_diffs = (design.unsqueeze(1) - design.unsqueeze(0)) ** 2.
x = theta[0]*design[:,0] + theta[1]*design[:,1]
phi0 = torch.tensor((1.,3.,0.1), dtype=torch.float32) ##Values used in paper

y = np.random.multivariate_normal(x, get_var_matrix(phi0, squared_diffs))
y = torch.tensor(y, dtype=torch.float32)

sample_phi(theta, phi0, y, design)
