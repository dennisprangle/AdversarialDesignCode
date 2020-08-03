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
opt_a = torch.optim.Adam(params=[dummy]) # Set lr=0. for SGD not GDA (i.e. FIG)
## Following two lines give a two time scale learning rate schedules
## following Heusel
#sched_e = torch.optim.lr_scheduler.LambdaLR(opt_e, lambda n : ((n+1) ** -0.9))
#sched_a = torch.optim.lr_scheduler.LambdaLR(opt_a, lambda n : ((n+1) ** -0.6))
## Following two lines use constant learning rates
sched_e = torch.optim.lr_scheduler.LambdaLR(opt_e, lambda n : 1.)
sched_a = torch.optim.lr_scheduler.LambdaLR(opt_a, lambda n : 1.)
optimizers = {'experimenter':opt_e, 'adversary':opt_a}
schedulers = {'experimenter':sched_e, 'adversary':sched_a}

init_design = np.random.uniform(0.,24.,15) # 15 samples from U[0,24]
init_design = np.sort(init_design) # Just so progress messages easier to read

advOpt = adv.AdvOpt(fim=fim,
                    optimizers=optimizers, schedulers=schedulers,
                    init_design_raw=init_design,
                    init_A_raw=np.zeros(fim.eta_dim()),
                    report_every=100, track_J=True)
advOpt.optimize(30000)
final_design = np.sort(advOpt.pointExchange())

########
##PLOTS
########

output = advOpt.stacked_output()

plt.figure()
out_design = output['design']
for i in range(out_design.shape[1]):
  plt.plot(output['iterations'], out_design[:,i])

plt.xlabel('Iterations')
plt.ylabel('Observation time')
plt.tight_layout()

plt.figure()
out_A = output['A']
out_A = out_A.reshape((out_A.shape[0], -1))
for i in range(out_A.shape[1]):
  plt.plot(output['iterations'], out_A[:,i])

plt.xlabel('Iterations')
plt.ylabel('A matrix')
plt.tight_layout()

plt.figure()
plt.plot(output['iterations'], output['objectiveK'])
plt.xlabel('Iterations')
plt.ylabel('K objective')
plt.yscale('symlog')
plt.tight_layout()

plt.figure()
plt.plot(output['iterations'], output['objectiveJ'])
plt.xlabel('Iterations')
plt.ylabel('J objective')
plt.tight_layout()

wait = input('Press enter to terminate')
