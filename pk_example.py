import torch
from torch.distributions import LogNormal
import numpy as np
import matplotlib.pyplot as plt
import advOpt as adv
plt.ion()

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
    # Unvectorised code for reference (theta is a vector)
    # x = 400. * theta[1] * (torch.exp(-theta[0]*design) - torch.exp(-theta[1]*design)) / (theta[2]*(theta[1]-theta[0]))
    # grad0 = x/(theta[1]-theta[0]) - design*400.*theta[1]/(theta[2]*(theta[1]-theta[0]))*torch.exp(-theta[0]*design)
    # grad1 = x/theta[1] - x/(theta[1]-theta[0]) + design*400.*theta[1]*torch.exp(-theta[1]*design)/(theta[2]*(theta[1]-theta[0]))
    # grad2 = -x/theta[2]
    # jacobian = torch.stack((grad0, grad1, grad2), dim=1)
    # fim = torch.mm(jacobian.transpose(0,1), jacobian)
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

dummy = torch.tensor(0.) #Because optimizer initialisation needs a target
opt_e = torch.optim.Adam(params=[dummy])
opt_a = torch.optim.Adam(params=[dummy])
sched_e = torch.optim.lr_scheduler.StepLR(opt_e, step_size=10**4, gamma=1)
sched_a = torch.optim.lr_scheduler.StepLR(opt_a, step_size=10**4, gamma=1)
optimizers = {'experimenter':opt_e, 'adversary':opt_a}
schedulers = {'experimenter':sched_e, 'adversary':sched_a}

advOpt = adv.AdvOpt(fim=fim,
                    optimizers=optimizers, schedulers=schedulers,
                    init_design_raw=np.random.uniform(0.,15.,10),
                    init_A_raw=np.zeros(fim.eta_dim()),
                    report_every=1000)
advOpt.optimize(50000)
