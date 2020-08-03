import torch
import numpy as np
import matplotlib.pyplot as plt
import advOpt as adv
plt.ion()

##############
##DEFINITIONS
##############

class geostats_FIM(adv.FIM):
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

  def estimateJ(self, design):
    """Return an estimate of J, the idealise objective

    In fact for this model, this calculate the exact value"""
    temp = self.estimate_expected_FIM(design)
    temp = torch.det(temp)
    return np.log(temp.detach().numpy())


def penalty(x, max_abs_x=0.5, magnitude=1e1):
  return magnitude * torch.sum(torch.relu(torch.abs(x) - max_abs_x))
  
fim = geostats_FIM(length_scale=0.05)

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

ndesign = 500
# ndesign samples from U[-0.5,0.5]^2
init_design = np.random.uniform(-0.5, 0.5, (ndesign,2))

advOpt = adv.AdvOpt(fim=fim,
                    optimizers=optimizers, schedulers=schedulers,
                    init_design_raw=init_design,
                    init_A_raw=np.zeros(fim.eta_dim()),
                    penalty=penalty,
                    report_every=10, track_J=True,
                    text_progress=False)
advOpt.optimize(1000)

########
##PLOTS
########

output = advOpt.stacked_output()
out_design = output['design']

plt.figure()
for i in range(out_design.shape[1]):
   plt.plot(out_design[:,i,0], out_design[:,i,1], '-')
plt.plot(out_design[-1,:,0], out_design[-1,:,1], 'o')
plt.tight_layout()

plt.figure()
plt.plot(out_design[-1,:,0], out_design[-1,:,1], 'o')
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
