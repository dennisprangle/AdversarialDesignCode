import torch
import numpy as np
import advOpt
import pickle

##############
##DEFINITIONS
##############

class geostats_FIM(advOpt.FIM):
  def __init__(self, length_scale=0.001, sigma1_squared=1., sigma2_squared=9.):
    self.length_scale_squared = np.power(length_scale, 2.)
    self.sigma1_squared = sigma1_squared
    self.sigma2_squared = sigma2_squared
    self.npars = 2

  def estimate_expected_FIM(self, design):
    """Return an estimate of the expected Fisher information

    In fact for this model this calculates the exact expected Fisher information"""
    ## Squared_distance matrix calculation could be more efficient
    ## See https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065
    diffs = design.unsqueeze(1) - design.unsqueeze(0)
    squared_distances =  torch.sum(diffs * diffs, -1)
    variance_matrix = self.sigma2_squared * \
                      torch.exp(-squared_distances / self.length_scale_squared) + \
                      self.sigma1_squared * torch.eye(design.shape[0], dtype=torch.float32)
    fim = torch.solve(design, variance_matrix)[0]
    fim = torch.mm(design.transpose(0,1), fim)
    return fim

  def initialise_J_estimation(self):
    """Initialise variables needed for estimating J

    For this model, nothing is done"""
    pass

  def estimateJ(self, design, adv=True):
    """Return an estimate of J objective

    In fact for this model, this calculates the exact value.

    `design` - which design to use
    `adv` - whether to calculate J_ADV (if True) or J_FIG (if False)
    """
    temp = self.estimate_expected_FIM(design)
    if adv:
      temp = torch.det(temp)
    else:
      temp = torch.trace(temp)
    return np.log(temp.detach().numpy())


def penalty(x, max_abs_x=0.5, magnitude=1e1):
  return magnitude * torch.sum(torch.relu(torch.abs(x) - max_abs_x), dim=(-2,-1))

##############
##OPTIMISATION
##############

def doOptimisation(length_scale=0.001, ndesign=500):
  fim = geostats_FIM(length_scale=length_scale)
  dummy = torch.tensor(0.) #Because optimizer initialisation needs a target
  opt_e = torch.optim.Adam(params=[dummy])
  opt_a = torch.optim.Adam(params=[dummy])
  ## Following two lines give a two time scale learning rate schedules
  ## following Heusel
  #sched_e = torch.optim.lr_scheduler.LambdaLR(opt_e, lambda n : ((n+1) ** -0.9))
  #sched_a = torch.optim.lr_scheduler.LambdaLR(opt_a, lambda n : ((n+1) ** -0.6))
  ## Following two lines use constant learning rates
  sched_e = torch.optim.lr_scheduler.LambdaLR(opt_e, lambda n : 1.)
  sched_a = torch.optim.lr_scheduler.LambdaLR(opt_a, lambda n : 1.)
  optimizers = {'experimenter':opt_e, 'adversary':opt_a}
  schedulers = {'experimenter':sched_e, 'adversary':sched_a}

  init_design = np.random.uniform(-0.5, 0.5, (ndesign,2))

  ad = advOpt.AdvOpt(fim=fim,
                     optimizers=optimizers, schedulers=schedulers,
                     init_design_raw=init_design,
                     init_A_raw=np.zeros(fim.eta_dim()),
                     penalty=penalty,
                     report_every=10,
                     text_progress=False,
                     track_J="ADV")
  ad.optimize(1000)
  return ad.stacked_output()

scales = [0.01, 0.02, 0.04, 0.08]
out = []
for s in scales:
  print("Scale ", s)
  out.append(doOptimisation(s))

to_save = zip(scales, out)
  
with open('geostats_example.pkl', 'wb') as outfile:
    pickle.dump(to_save, outfile)
