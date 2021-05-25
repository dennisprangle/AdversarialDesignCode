import torch
import numpy as np
import pickle
import argparse
import advOpt

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

    In fact for this model this calculates the exact expected Fisher information

    `design` is a tensor of designs, batched along dimension 0

    The output is a tensor of shape `design.shape[0:2] + (self.npars, self.npars)`
    """
    ## squared_distances matrix calculation could be more efficient
    ## See https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065
    diffs = design.unsqueeze(2) - design.unsqueeze(1)
    squared_distances =  torch.sum(diffs ** 2., -1)
    variance_matrix = self.sigma2_squared * \
                      torch.exp(-squared_distances / self.length_scale_squared) + \
                      self.sigma1_squared * torch.eye(design.shape[1], dtype=torch.float32).unsqueeze(0)
    fim = torch.solve(design, variance_matrix)[0]
    fim = torch.matmul(design.transpose(1,2), fim)
    return fim

  def initialise_J_estimation(self):
    """Initialise variables needed for estimating J

    For this model, nothing is done"""
    pass

  def estimateJ(self, design, adv=True):
    """Return an estimate of J objective

    In fact for this model, this calculates the exact value.

    `design` is a tensor of designs, batched along dimension 0  
    `adv` - whether to calculate J_ADV (if True) or J_FIG (if False)

    The output is a tensor of shape `design.shape[0]`
    """
    temp = self.estimate_expected_FIM(design)
    if adv:
      temp = torch.det(temp)
    else:
      temp = temp.diagonal(dim1=1,dim2=2).sum(dim=1)
    return temp


def penalty(x, max_abs_x=0.5, magnitude=1e1):
  return magnitude * torch.sum(torch.relu(torch.abs(x) - max_abs_x), dim=(-2,-1))

##############
##OPTIMISATION
##############

def main(ndesign, gda_its, length_scale, ttur, show_progress, seed, name):
  torch.manual_seed(seed)
  fim = geostats_FIM(length_scale=length_scale)
  dummy = torch.tensor(0.) #Because optimizer initialisation needs a target
  opt_e = torch.optim.Adam(params=[dummy])
  opt_a = torch.optim.Adam(params=[dummy])
  if ttur == True:
    sched_e = torch.optim.lr_scheduler.LambdaLR(opt_e, lambda n : ((n+1) ** -0.9))
    sched_a = torch.optim.lr_scheduler.LambdaLR(opt_a, lambda n : ((n+1) ** -0.6))
  else:
    sched_e = torch.optim.lr_scheduler.LambdaLR(opt_e, lambda n : 1.)
    sched_a = torch.optim.lr_scheduler.LambdaLR(opt_a, lambda n : 1.)
  optimizers = {'experimenter':opt_e, 'adversary':opt_a}
  schedulers = {'experimenter':sched_e, 'adversary':sched_a}

  init_design = np.random.uniform(-0.5, 0.5, (1,ndesign,2))
  init_A_raw = np.zeros((1, fim.eta_dim()))

  ad = advOpt.AdvOpt(fim=fim,
                     optimizers=optimizers, schedulers=schedulers,
                     init_design_raw=init_design,
                     init_A_raw=init_A_raw,
                     penalty=penalty,
                     report_every=10,
                     text_progress=show_progress,
                     track_J="ADV")
  ad.optimize(gda_its)
  output = ad.stacked_output()

  file_name = "./outputs/" + name + ".pkl"
  with open(file_name, 'wb') as outfile:
    pickle.dump(output, outfile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gradient descent ascent optimal design for a geostatistical regression")
    parser.add_argument("--ndesign", default=500, type=int)
    parser.add_argument("--gda-iterations", default=1000, type=int)
    parser.add_argument("--length-scale", default=0.001, type=float)
    parser.add_argument('--ttur', dest='ttur', action='store_true')
    parser.set_defaults(ttur=False)
    parser.add_argument('--show-progress', dest='show_progress', action='store_true')
    parser.set_defaults(show_progress=False)
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--name", default="geostats_example", type=str)
    args = parser.parse_args()
    main(args.ndesign, args.gda_iterations, args.length_scale, args.ttur,
         args.show_progress, args.seed, args.name)
