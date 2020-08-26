import torch
import numpy as np
from time import time

class FIM:
  """Fisher information matrix details"""
  def __init__(self):
    ##Following should be set by subclasses if needed
    self.npars = None ## Positive integer: used in makeA (always needed)
    self.nsamples = None ## Positive integer: used in estimate_expected_FIM
    self.prior = None ## PyTorch distribution: used in estimate_expected_FIM

  def estimate_FIM(self, theta, design):
    """Return estimates of Fisher information matrices

    `theta` is a `self.nsamples` x `self.npars` tensor of parameters
    `design` is a single design

    The output should be a `self.nsamples` x `self.npars` x `self.npars` tensor
    """
    raise NotImplementedError

  def estimate_expected_FIM(self, design):
    """Return an estimate of the expected Fisher information"""
    ## Default code performs Monte Carlo estimation.
    ## Subclasses can override and use other approaches if desired.
    theta = self.prior.sample((self.nsamples,))
    fim = self.estimate_FIM(theta, design)
    return torch.mean(fim, dim=0)

  def estimateK(self, design, A):
    """Return an unbiased estimate of K, the adversarial design objective"""
    temp = self.estimate_expected_FIM(design)
    temp = torch.mm(A, temp)
    temp = torch.mm(temp, A.transpose(0,1))
    return -temp.trace()

  def initialise_J_estimation(self):
    """Initialise variables needed for estimating J"""
    self.thetas_for_J = self.prior.sample((1000,))

  def estimateJ(self, design, adv=True):
    """Return an estimate of J objective

    `design` - which design to use
    `adv` - whether to calculate J_ADV (if True) or J_FIG (if False)
    """
    temp = self.estimate_FIM(self.thetas_for_J, design)
    temp = torch.mean(temp, dim=0)
    if adv:
      temp = torch.det(temp)
    else:
      temp = torch.trace(temp)
    return np.log(temp.detach().numpy())

  def eta_dim(self):
    """Number of entries required in eta vector

    i.e. count of non-zero entries in `npars` x `npars` triangular matrix minus 1
    """
    return self.npars * (self.npars + 1) // 2 - 1

  def makeA(self, x):
    """Convert tensor `x` to tensor of `A` matrices
    
    The final dimension of `x` should be `npars(npars+1)/2 - 1`

    Returns `A`, a tensor with shape `x.shape[:-1] + (npars, npars)`
    consisting of lower triangular matrices with positive diagonal
    and bottom right element set to give determinant 1.
    The elements of `x` fill the remainder of the matrix in an
    anticlockwise spiral.
    See `math.fill_triangular` from tensorflow probability for more details:
    the code here uses the same mathematical approach.
    """
    z = torch.zeros(x.shape[:-1] + (1,), dtype=x.dtype)
    x = torch.cat([z,x], dim=-1)
    x_list = [x[..., self.npars:], x.flip(-1)]
    A = torch.reshape(torch.cat(x_list, dim=-1), x.shape[:-1] + (self.npars, self.npars))
    A = torch.tril(A)
    d = A.diagonal(dim1=-2, dim2=-1)
    A[...,-1,-1] = -d.sum(-1)
    d.exp_()
    return A

  
class AdvOpt:
  """Optimises a Bayesian experimental design problem"""

  def __init__(self, fim,
               optimizers, schedulers,
               init_design_raw, init_A_raw,
               make_design=lambda x:x,
               penalty = lambda x:0.,
               report_every=500, text_progress=True,
               track_J=None):
    """
    `fim` - object of `FIM` class
    `optimizers` - dict of optimizers for `experimenter` and `adversary`
    (with no params)
    `schedulers` - dict of schedulers for `experimenter` and `adversary`
    `init_design_raw` - initial values for `design_raw` (tuple, list or array)
    `init_A_raw` - initial values for variables controlling A matrix
    (tuple, list or array)
    `make_design` - function mapping a `design_raw` tensor to design
    `penalty` - function mapping a `design` tensor to a penalty
    `report_every` - how often to record/report progress
    `text_progress` - report text summaries of progress if `True`
    `track_J` - can be "ADV", "FIG" or None
    """
    self.start_time = time()
    self.report_every = report_every
    self.track_J = track_J
    self.text_progress = text_progress
    if text_progress == True:
      np.set_printoptions(precision=3)

    self.fim = fim
    self.make_design = make_design
    self.penalty = penalty
    
    self.design_raw = torch.tensor(init_design_raw, dtype=torch.float32,
                                   requires_grad=True)
    eta_dim = fim.eta_dim()
    if len(init_A_raw) != eta_dim:
      raise ValueError("init_A_raw wrong length, should be " + str(eta_dim))
    self.A_raw = torch.tensor(init_A_raw, dtype=torch.float32,
                                   requires_grad=True)
    self.optimizers = optimizers
    self.schedulers = schedulers
    self.optimizers['experimenter'].add_param_group({'params': self.design_raw})
    self.optimizers['adversary'].add_param_group({'params': self.A_raw})

    self.output = {'iterations':[], 'time':[], 'design':[], 'A':[],
                   'objectiveK':[]}
    if track_J is not None:
      self.fim.initialise_J_estimation()
      self.output['objectiveJ'] = []
    self.iteration = 0
    
    
  def optimize(self, iterations=10**5):
    """Find optimal design numerically"""
    for i in range(iterations):
      for opt in self.optimizers.values():
        opt.zero_grad()
      A = self.fim.makeA(self.A_raw)
      design = self.make_design(self.design_raw)
      objective = self.fim.estimateK(design, A) + self.penalty(design)
      objective.backward()
      self.A_raw.grad *= -1. # Do ascent instead of descent
      self.optimizers['experimenter'].step()
      self.schedulers['experimenter'].step()
      self.optimizers['adversary'].step()
      self.schedulers['adversary'].step()
      self.iteration += 1
      if (self.iteration % self.report_every) == 0:
        elapsed = time() - self.start_time
        self.output['time'].append(elapsed)
        self.output['iterations'].append(self.iteration)
        design_np = design.detach().numpy().copy()
        A_np = A.detach().numpy().copy()
        self.output['design'].append(design_np)
        self.output['A'].append(A_np)
        self.output['objectiveK'].append(float(objective))
        if self.track_J == "ADV":
          self.output['objectiveJ'].append(float(self.fim.estimateJ(design, adv=True)))
        elif self.track_J == "FIG":
          self.output['objectiveJ'].append(float(self.fim.estimateJ(design, adv=False)))

        if self.text_progress:
          print("Iteration {:d}, time (mins) {:.1f}, K objective {:.2f}".\
                format(i+1, elapsed/60, float(objective)))
          print("Design:\n", design_np)
          print("A matrix:\n", A_np)
          print("Learning rate: experimenter {:.6f} adversary {:.6f}\n".\
                format(self.optimizers['experimenter'].param_groups[0]['lr'],
                       self.optimizers['adversary'].param_groups[0]['lr']))


  def pointExchange(self, maxits=10, adv=True):
    """Point exchange optimisation to fine tune design

    Currently this requires `self.design` to be a vector

    `maxits` is maximum number of iterations to use
    `adv` is True for the J_ADV objective or False for J_FIG objective
    """
    design = self.make_design(self.design_raw).detach().numpy().copy()
    bestJ = self.fim.estimateJ(torch.tensor(design), adv)
    best_new_design = design.copy()
    for outer_counter in range(maxits):
      improvement = False
      for i in range(design.shape[0]):
        for j in range(design.shape[0]):
          if i==j:
            continue
          proposed_design = design.copy()
          proposed_design[i] = design[j]
          newJ = self.fim.estimateJ(torch.tensor(proposed_design), adv)
          if (newJ > bestJ):
            bestJ = newJ
            best_new_design = proposed_design
            improvement = True

      design = best_new_design

      if self.text_progress==True:
        print("Point exchange iteration {:d}".format(outer_counter+1))
        print("Design:\n", np.sort(proposed_design))
        print("Objective J estimate: {:.3f}\n".format(bestJ))

      if improvement == False:
        break

    return design


  def stacked_output(self):
    """Return optimisation output stacked into arrays"""
    stacked_output = {}
    for key in self.output.keys():
      stacked_output[key] = np.stack(self.output[key])

    return stacked_output
