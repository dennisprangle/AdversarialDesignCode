import torch
import numpy as np
import matplotlib.pyplot as plt
import advOpt as adv
plt.ion()

##############
##DEFINITIONS
##############

class Poisson_FIM(adv.FIM):
  def __init__(self, omega):
    self.omega = omega
    self.npars = 2

  def estimate_expected_FIM(self, design):
    """Return an estimate of the expected Fisher information

    In fact for this model this calculates the exact expected Fisher information"""
    fim = torch.zeros((2,2), dtype=torch.float32)
    fim[0,0] = design[0]*self.omega[0]
    fim[1,1] = (1-design[0])*self.omega[1]
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

fim = Poisson_FIM(omega=(2., 1.))

##############
##VECTOR FIELD
##############

logit_design_grid, eta11_grid = np.meshgrid(np.linspace(-0.65, 0.65, 10),
                                            np.linspace(-0.2, -0.14, 11))
logit_design_grad_grid = np.zeros_like(logit_design_grid)
eta11_grad_grid = np.zeros_like(eta11_grid)
for i in range(eta11_grid.shape[0]):
  for j in range(eta11_grid.shape[1]):
    logit_design = torch.tensor([logit_design_grid[i,j]], requires_grad=True)
    eta11 = torch.tensor(eta11_grid[i,j], requires_grad=True)
    eta = torch.zeros(2, dtype=torch.float32)
    eta[1] += eta11
    A = fim.makeA(eta)
    design = torch.sigmoid(logit_design)
    objective = fim.estimateK(design, A)
    objective.backward()
    logit_design_grad_grid[i,j] = float(logit_design.grad)
    eta11_grad_grid[i,j] = float(eta11.grad)

vector_field_plot = plt.quiver(logit_design_grid, eta11_grid,
                               -logit_design_grad_grid, eta11_grad_grid/1000,
                               angles='xy')

vector_field_plot.axes.set_xlabel(r'$\lambda$')
vector_field_plot.axes.set_ylabel(r'$\eta_{11}$')

##############
##OPTIMISATION
##############

dummy = torch.tensor(0.) #Because optimizer initialisation needs a target
opt_e = torch.optim.SGD(params=[dummy], lr=1e-2)
opt_a = torch.optim.SGD(params=[dummy], lr=1e-3)
sched_e = torch.optim.lr_scheduler.StepLR(opt_e, step_size=10**4, gamma=1)
sched_a = torch.optim.lr_scheduler.StepLR(opt_a, step_size=10**4, gamma=1)
optimizers = {'experimenter':opt_e, 'adversary':opt_a}
schedulers = {'experimenter':sched_e, 'adversary':sched_a}

ad1 = adv.AdvOpt(fim=fim, make_design=torch.sigmoid,
                    optimizers=optimizers, schedulers=schedulers,
                    init_design_raw=[-0.2],
                    init_A_raw=(0., -0.15),
                    report_every=5, text_progress=False)
ad1.optimize(25000)
output1 = ad1.stacked_output()

opt_e = torch.optim.SGD(params=[dummy], lr=1e-2)
opt_a = torch.optim.SGD(params=[dummy], lr=1e-4)
sched_e = torch.optim.lr_scheduler.StepLR(opt_e, step_size=10**4, gamma=1)
sched_a = torch.optim.lr_scheduler.StepLR(opt_a, step_size=10**4, gamma=1)
optimizers = {'experimenter':opt_e, 'adversary':opt_a}
schedulers = {'experimenter':sched_e, 'adversary':sched_a}

ad2 = adv.AdvOpt(fim=fim, make_design=torch.sigmoid,
                    optimizers=optimizers, schedulers=schedulers,
                    init_design_raw=[-0.2],
                    init_A_raw=(0., -0.15),
                    report_every=5, text_progress=False)
ad2.optimize(25000)
output2 = ad2.stacked_output()

opt_e = torch.optim.SGD(params=[dummy], lr=1e-2)
opt_a = torch.optim.SGD(params=[dummy], lr=1e-5)
sched_e = torch.optim.lr_scheduler.StepLR(opt_e, step_size=10**4, gamma=1)
sched_a = torch.optim.lr_scheduler.StepLR(opt_a, step_size=10**4, gamma=1)
optimizers = {'experimenter':opt_e, 'adversary':opt_a}
schedulers = {'experimenter':sched_e, 'adversary':sched_a}

ad3 = adv.AdvOpt(fim=fim, make_design=torch.sigmoid,
                    optimizers=optimizers, schedulers=schedulers,
                    init_design_raw=[-0.2],
                    init_A_raw=(0., -0.15),
                    report_every=5, text_progress=False)
ad3.optimize(25000)
output3 = ad3.stacked_output()

logit = lambda x: np.log(x/(1-x))
dot_indices = range(99, 5000, 100)
toplot = [(output1, 'b'), (output2, 'g'), (output3, 'r')]
for (o, col) in toplot:
  plt.plot(logit(o['design'][dot_indices,0]),
           np.log(o['A'][dot_indices,0,0]), 'o' + col)
  plt.plot(logit(o['design'][:,0]), np.log(o['A'][:,0,0]), '-' + col)

plt.xlim([-0.7, 0.7])
plt.ylim([-0.203, -0.137])

plt.tight_layout()
plt.savefig('poisson_vector_field.pdf')

## Uncoment the following for traceplots

# plt.figure()
# for (o, _) in toplot:
#   plt.plot(o['iterations'], logit(o['design'][:,0]))

# plt.ylim([-0.7, 0.7])
# plt.xlabel('Iterations')
# plt.ylabel(r'$\lambda$')
# plt.tight_layout()
# plt.savefig('poisson_traceplot_design.pdf')

# plt.figure()
# for (o, _) in toplot:
#   plt.plot(o['iterations'], np.log(o['A'][:,0,0]))

# plt.ylim([-0.205, -0.135])
# plt.xlabel('Iterations')
# plt.ylabel(r'$\eta_{11}$')
# plt.tight_layout()
# plt.savefig('poisson_traceplot_param.pdf')

# wait = input('Press enter to terminate')
