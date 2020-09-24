import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle
import argparse
import advOpt

##############
##DEFINITIONS
##############

class Poisson_FIM(advOpt.FIM):
  def __init__(self, omega):
    self.omega = omega
    self.npars = 2

  def estimate_expected_FIM(self, design):
    """Return an estimate of the expected Fisher information

    In fact for this model this calculates the exact expected Fisher information

    `design` is a tensor of designs, batched along dimension 0

    The output is a `design.shape[0]` x `self.npars` x `self.npars` tensor
    """
    fim = torch.zeros((design.shape[0],2,2), dtype=torch.float32)
    fim[:,0,0] = design[:,0]*self.omega[0]
    fim[:,1,1] = (1-design[:,0])*self.omega[1]
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

def main(gda_its, lr_e, lr_a, init_design_raw, init_A_raw, show_progress, name):
  fim = Poisson_FIM(omega=(2., 1.))
  dummy = torch.tensor(0.) #Because optimizer initialisation needs a target
  opt_e = torch.optim.SGD(params=[dummy], lr=lr_e)
  opt_a = torch.optim.SGD(params=[dummy], lr=lr_a)
  sched_e = torch.optim.lr_scheduler.StepLR(opt_e, step_size=10**4, gamma=1)
  sched_a = torch.optim.lr_scheduler.StepLR(opt_a, step_size=10**4, gamma=1)
  optimizers = {'experimenter':opt_e, 'adversary':opt_a}
  schedulers = {'experimenter':sched_e, 'adversary':sched_a}

  init_design_raw = np.array(init_design_raw, ndmin=2)
  init_A_raw = np.array(init_A_raw, ndmin=2)
  ad = advOpt.AdvOpt(fim=fim, make_design=torch.sigmoid,
                      optimizers=optimizers, schedulers=schedulers,
                      init_design_raw=init_design_raw,
                      init_A_raw=init_A_raw,
                      report_every=5, text_progress=show_progress)
  ad.optimize(gda_its)
  output = ad.stacked_output()
  file_name = "./outputs/" + name + ".pkl"
  with open(file_name, 'wb') as outfile:
    pickle.dump(output, outfile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gradient descent ascent optimal design for a Poisson model")
    parser.add_argument("--gda-iterations", default=25000, type=int)
    parser.add_argument("--lr-e", default=1e-2, type=float)
    parser.add_argument("--lr-a", default=1e-4, type=float)
    parser.add_argument("--init_design_raw", default=-0.2, type=float)
    parser.add_argument("--init_A_raw", nargs="+", default=[0., -0.15], type=float)
    parser.add_argument('--show-progress', dest='show_progress', action='store_true')
    parser.set_defaults(show_progress=False)
    parser.add_argument("--name", default="pk_example", type=str)
    args = parser.parse_args()
    main(args.gda_iterations, args.lr_e, args.lr_a, args.init_design_raw,
         args.init_A_raw, args.show_progress, args.name)
