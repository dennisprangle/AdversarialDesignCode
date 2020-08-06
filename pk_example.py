import torch
from torch.distributions import LogNormal
import numpy as np
import advOpt as adv
from time import time
import pickle

##############
##DEFINITIONS
##############

class PK_FIM(adv.FIM):
  def __init__(self, nsamples):
    self.npars = 3
    self.nsamples = nsamples
    loc = torch.tensor(np.log((0.1, 1., 20.)), dtype=torch.float32)
    scale = torch.tensor(np.sqrt((0.05, 0.05, 0.05)), dtype=torch.float32)
    self.prior = LogNormal(loc, scale)

  def estimate_FIM(self, theta, design):
    design = design.unsqueeze(0)
    theta1 = theta[:, 0:1]
    theta2 = theta[:, 1:2]
    theta3 = theta[:, 2:3]
    x = 400. * theta2 * (torch.exp(-theta1*design) - torch.exp(-theta2*design)) / (theta3*(theta2-theta1))    
    grad0 = x/(theta2-theta1) - design*400.*theta2/(theta3*(theta2-theta1))*torch.exp(-theta1*design)
    grad1 = x/theta2 - x/(theta2-theta1) + design*400.*theta2*torch.exp(-theta2*design)/(theta3*(theta2-theta1))
    grad2 = -x/theta3
    jacobian = torch.stack((grad0, grad1, grad2), dim=2)
    fim = torch.matmul(jacobian.transpose(1,2), jacobian)
    return fim


fim = PK_FIM(nsamples=100)

##############
##OPTIMISATION
##############

def doOptimisation(init_design, SGD=False):
  dummy = torch.tensor(0.) #Because optimizer initialisation needs a target
  opt_e = torch.optim.Adam(params=[dummy])
  if SGD:
    opt_a = torch.optim.Adam(params=[dummy], lr=0.) # SGD (no update of A)
  else:
    opt_a = torch.optim.Adam(params=[dummy]) # GDA
  ## Following two lines give a two time scale learning rate schedules
  ## following Heusel
  #sched_e = torch.optim.lr_scheduler.LambdaLR(opt_e, lambda n : ((n+1) ** -0.9))
  #sched_a = torch.optim.lr_scheduler.LambdaLR(opt_a, lambda n : ((n+1) ** -0.6))
  ## Following two lines use constant learning rates
  sched_e = torch.optim.lr_scheduler.LambdaLR(opt_e, lambda n : 1.)
  sched_a = torch.optim.lr_scheduler.LambdaLR(opt_a, lambda n : 1.)
  optimizers = {'experimenter':opt_e, 'adversary':opt_a}
  schedulers = {'experimenter':sched_e, 'adversary':sched_a}

  advOpt = adv.AdvOpt(fim=fim,
                      optimizers=optimizers, schedulers=schedulers,
                      init_design_raw=init_design,
                      init_A_raw=np.zeros(fim.eta_dim()),
                      report_every=100, track_J=True,
                      text_progress=False)
  advOpt.optimize(30000)
  output = advOpt.stacked_output()
  start_time = time()
  final_design = np.sort(advOpt.pointExchange())
  point_exchange_time = time() - start_time
  return output, final_design, point_exchange_time


nreps = 100
init_designs = np.random.uniform(0.,24.,(nreps, 15)) # 15 samples from U[0,24]
init_designs = np.sort(init_designs, axis=1) # So progress messages read easier
out_GDA = []
out_SGD = []
for i in range(nreps):
  print("Iteration ", i+1)
  out_GDA.append(doOptimisation(init_designs[i,:]))
  out_SGD.append(doOptimisation(init_designs[i,:], SGD=True))

out = (out_GDA, out_SGD)

with open('pk_example.pkl', 'wb') as outfile:
    pickle.dump(out, outfile)
