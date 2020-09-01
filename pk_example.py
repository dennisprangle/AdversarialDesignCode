import torch
from torch.distributions import LogNormal
import numpy as np
from time import time
import pickle
import argparse
import advOpt

##############
##DEFINITIONS
##############

class PK_FIM(advOpt.FIM):
  def __init__(self, nsamples):
    self.npars = 3
    self.nsamples = nsamples
    loc = torch.tensor(np.log((0.1, 1., 20.)), dtype=torch.float32)
    scale = torch.tensor(np.sqrt((0.05, 0.05, 0.05)), dtype=torch.float32)
    self.prior = LogNormal(loc, scale)

  def estimate_FIM(self, theta, design):
    design = design.unsqueeze(1)
    theta1 = theta[:, 0:1].unsqueeze(0)
    theta2 = theta[:, 1:2].unsqueeze(0)
    theta3 = theta[:, 2:3].unsqueeze(0)
    x = 400. * theta2 * (torch.exp(-theta1*design) - torch.exp(-theta2*design)) / (theta3*(theta2-theta1))
    grad0 = x/(theta2-theta1) - design*400.*theta2/(theta3*(theta2-theta1))*torch.exp(-theta1*design)
    grad1 = x/theta2 - x/(theta2-theta1) + design*400.*theta2*torch.exp(-theta2*design)/(theta3*(theta2-theta1))
    grad2 = -x/theta3
    jacobian = torch.stack((grad0, grad1, grad2), dim=3)
    fim = torch.matmul(jacobian.transpose(2,3), jacobian)
    return fim


##############
##OPTIMISATION
##############

def main(gda_its, nsamples, nparallel, sgd, ttur, show_progress, point_exchange,
         seed, name):
  torch.manual_seed(seed)
  fim = PK_FIM(nsamples)
  dummy = torch.tensor(0.) #Because optimizer initialisation needs a target
  opt_e = torch.optim.Adam(params=[dummy])
  if sgd == True:
    opt_a = torch.optim.Adam(params=[dummy], lr=0.) # i.e. no update of A
  else:
    opt_a = torch.optim.Adam(params=[dummy])
  if ttur == True:
    sched_e = torch.optim.lr_scheduler.LambdaLR(opt_e, lambda n : ((n+1) ** -0.9))
    sched_a = torch.optim.lr_scheduler.LambdaLR(opt_a, lambda n : ((n+1) ** -0.6))
  else:
    sched_e = torch.optim.lr_scheduler.LambdaLR(opt_e, lambda n : 1.)
    sched_a = torch.optim.lr_scheduler.LambdaLR(opt_a, lambda n : 1.)
  optimizers = {'experimenter':opt_e, 'adversary':opt_a}
  schedulers = {'experimenter':sched_e, 'adversary':sched_a}

  init_A_raw = np.zeros((nparallel, fim.eta_dim()))
  init_design = np.random.uniform(0.,24.,(nparallel, 15))
  if show_progress == True:
    init_design = np.sort(init_design, axis=1) # Helps readability of messages

  track_J = "FIG" if sgd else "ADV"
  ad = advOpt.AdvOpt(fim=fim,
                     optimizers=optimizers, schedulers=schedulers,
                     init_design_raw=init_design, init_A_raw=init_A_raw,
                     report_every=100, track_J=track_J,
                     text_progress=show_progress)
  ad.optimize(gda_its)
  output = ad.stacked_output()
  start_time = time()
  if point_exchange == True:
    final_design = np.sort(ad.pointExchange(adv=not(sgd)), axis=1)
    output['final_design'] = final_design
    output['point_exchange_time'] = time() - start_time

  file_name = "./outputs/" + name + ".pkl"
  with open(file_name, 'wb') as outfile:
    pickle.dump(output, outfile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gradient descent ascent optimal design for a pharmacokinetic model")
    parser.add_argument("--nsamples", default=100, type=int)
    parser.add_argument("--gda-iterations", default=30000, type=int)
    parser.add_argument('--sgd', dest='sgd', action='store_true')
    parser.set_defaults(sgd=False)
    parser.add_argument('--ttur', dest='ttur', action='store_true')
    parser.set_defaults(ttur=False)
    parser.add_argument("--nparallel", default=10, type=int)
    parser.add_argument('--show-progress', dest='show_progress', action='store_true')
    parser.set_defaults(show_progress=False)
    parser.add_argument('--point-exchange', dest='point_exchange', action='store_true')
    parser.set_defaults(point_exchange=True)
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--name", default="pk_example", type=str)
    args = parser.parse_args()
    main(args.gda_iterations, args.nsamples, args.nparallel, args.sgd, args.ttur, 
         args.show_progress, args.point_exchange, args.seed, args.name)
