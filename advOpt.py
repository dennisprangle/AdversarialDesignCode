import torch
import numpy as np

class FIM:
  """Fisher information matrix details"""
  def __init__(self):
    ##Following should be set by subclasses if needed
    self.npars = None ## Positive integer: used in makeA (always needed)
    self.nsamples = None ## Positive integer: used in estimate_expected_FIM
    self.prior = None ## PyTorch distribution: used in estimate_expected_FIM

  def set_advOpt(self, advOpt):
    """Set the AdvOpt object this is embedded within.
    This allows complicated FIM subclasses to access its details."""
    self.advOpt = advOpt

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
    return torch.sum(fim, dim=0)

  def estimateK(self, design, A):
    """Return an unbiased estimate of K, the adversarial design objective"""
    temp = self.estimate_expected_FIM(design)
    temp = torch.mm(A, temp)
    temp = torch.mm(temp, A.transpose(0,1))
    return -temp.trace()

  def makeA(self, x):
    """Convert vector `x` to a valid `A` matrix
    
    `x` should be a vector of length `npars(npars+1)/2 - 1`

    Returns `A`, a `npars`x`npars` lower triangular matrix
    with positive diagonal and bottom right element set to give determinant 1.
    The elements of `x` fill the remainder of the matrix in an
    anticlockwise spiral.
    See `math.fill_triangular` from tensorflow probability for more details:
    the code here uses the same mathematical approach.
    """
    z = torch.tensor([0.], dtype=x.dtype)
    x = torch.cat([z,x])
    x_list = [x[self.npars:], torch.flip(x, [0])]
    A = torch.reshape(torch.cat(x_list), (self.npars, self.npars))
    A = torch.tril(A)
    d = A.diagonal()
    A[-1,-1] = -torch.sum(d)
    d.exp_()
    return A

  
class AdvOpt:
  """Optimises a Bayesian experimental design problem"""

  def __init__(self, fim,
               optimizers, schedulers,
               init_design_raw, init_A_raw,
               make_design=lambda x:x,
               report_every=500, text_progress=True):
    """
    `fim` - object of `FIM` class
    `optimizers` - dict of optimizers for `experimenter` and `adversary`
    (with no params)
    `schedulers` - dict of schedulers for `experimenter` and `adversary`
    `init_design_raw` - initial values for `design_raw` (tuple, list or array)
    `init_A_raw` - initial values for variables controlling A matrix
    (tuple, list or array)
    `make_design` - function mapping a `design_raw` vector to design
    `report_every` - how often to record/report progress
    `text_progress` - report text summaries of progress if `True`
    """
    self.report_every = report_every
    self.text_progress = text_progress

    self.fim = fim
    self.fim.set_advOpt(self)
    self.make_design = make_design
    
    self.design_raw = torch.tensor(init_design_raw, dtype=torch.float32,
                                   requires_grad=True)
    adim = fim.npars * (fim.npars + 1) / 2 - 1 ## i.e. number of non-zero entries
                                               ## in triangular matrix, minus 1
    if len(init_A_raw) != adim:
      raise ValueError("init_A_raw wrong length, should be " + str(adim))    
    self.A_raw = torch.tensor(init_A_raw, dtype=torch.float32,
                                   requires_grad=True)
    self.optimizers = optimizers
    self.schedulers = schedulers
    self.optimizers['experimenter'].add_param_group({'params': self.design_raw})
    self.optimizers['adversary'].add_param_group({'params': self.A_raw})

    self.output = {'iterations':[], 'design':[], 'A':[], 'objective':[]}
    self.iteration = 0
    
    
  def optimize(self, iterations=10**5):
    """Find optimal design numerically"""
    for i in range(iterations):
      for opt in self.optimizers.values():
        opt.zero_grad()
      A = self.fim.makeA(self.A_raw)
      design = self.make_design(self.design_raw)
      objective = self.fim.estimateK(design, A)
      objective.backward()
      self.A_raw.grad *= -1. # Do ascent instead of descent
      self.optimizers['experimenter'].step()
      self.schedulers['experimenter'].step()
      self.optimizers['adversary'].step()
      self.schedulers['adversary'].step()
      self.iteration += 1
      if (self.iteration % self.report_every) == 0:
        self.output['iterations'].append(self.iteration)
        design_np = design.detach().numpy().copy()
        A_np = A.detach().numpy().copy()
        self.output['design'].append(design_np)
        self.output['A'].append(A_np)
        self.output['objective'].append(float(objective))
        if self.text_progress:
          print("Iteration {:d}, objective {:.2f}".\
                format(i+1, float(objective)))
          print("design ", design_np)
          print("A ", A_np)
          print("Learning rate: experimenter {:.6f} adversary {:.6f}".\
                format(self.optimizers['experimenter'].param_groups[0]['lr'],
                       self.optimizers['adversary'].param_groups[0]['lr']))


  def stacked_output(self):
    """Return optimisation output stacked into arrays"""
    stacked_output = {}
    for key in self.output.keys():
      stacked_output[key] = np.stack(self.output[key])

    return stacked_output
