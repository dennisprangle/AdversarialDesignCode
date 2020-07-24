import torch
import numpy as np
import matplotlib.pyplot as plt
import advDesign as adv
plt.ion()

class Poisson_FIM(adv.FIM):
  def __init__(self, omega):
    self.omega = omega
    self.npars = 2

  def estimate(self, design):
    fim = torch.zeros((2,2), dtype=torch.float32)
    fim[0,0] = design[0]*self.omega[0]
    fim[1,1] = (1-design[0])*self.omega[1]
    return fim

def make_design_Poisson(x):
  return torch.sigmoid(x)

exampleFIM = Poisson_FIM(omega=(2., 1.))

##############
##VECTOR FIELD
##############

design_grid, A11_grid = np.meshgrid(np.linspace(0.35, 0.7, 10),
                                    np.linspace(0.82, 0.87, 10))
design_grad_grid = np.zeros_like(design_grid)
A11_grad_grid = np.zeros_like(A11_grid)
for i in range(10):
  for j in range(10):
    design = torch.tensor([design_grid[i,j]], requires_grad=True)
    A11 = torch.tensor(A11_grid[i,j], requires_grad=True)
    A_raw = torch.zeros(2, dtype=torch.float32)
    A_raw[1] += torch.log(A11)
    A = exampleFIM.makeA(A_raw)
    logit_design = torch.log(design / (1.-design))
    objective = exampleFIM.estimateK(logit_design, A)
    objective.backward()
    design_grad_grid[i,j] = float(design.grad)
    A11_grad_grid[i,j] = float(A11.grad)

vector_field_plot = plt.quiver(design_grid, A11_grid,
                               -design_grad_grid, A11_grad_grid/100,
                               angles='xy')

vector_field_plot.axes.set_xlabel(r'$\tau$')
vector_field_plot.axes.set_ylabel(r'$A_{11}$')

##############
##OPTIMISATION
##############

dummy = torch.tensor(0.) #Because optimizer need some tensor to initialise
opt_e = torch.optim.SGD(params=[dummy], lr=1e-2)
opt_a = torch.optim.SGD(params=[dummy], lr=1e-3)
sched_e = torch.optim.lr_scheduler.StepLR(opt_e, step_size=10**4, gamma=1)
sched_a = torch.optim.lr_scheduler.StepLR(opt_a, step_size=10**4, gamma=1)
optimizers = {'experimenter':opt_e, 'adversary':opt_a}
schedulers = {'experimenter':sched_e, 'adversary':sched_a}

ad1 = adv.AdvDesign(fim=exampleFIM, make_design=make_design_Poisson,
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

ad2 = adv.AdvDesign(fim=exampleFIM, make_design=make_design_Poisson,
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

ad3 = adv.AdvDesign(fim=exampleFIM, make_design=make_design_Poisson,
                    optimizers=optimizers, schedulers=schedulers,
                    init_design_raw=[-0.2],
                    init_A_raw=(0., -0.15),
                    report_every=5, text_progress=False)
ad3.optimize(25000)
output3 = ad3.stacked_output()

dot_indices = range(99, 5000, 100)
plt.plot(output3['design'][dot_indices,0],
         output3['A'][dot_indices,0,0], 'or')
plt.plot(output2['design'][dot_indices,0],
         output2['A'][dot_indices,0,0], 'og')
plt.plot(output1['design'][dot_indices,0],
         output1['A'][dot_indices,0,0], 'ob')
plt.plot(output3['design'][:,0], output3['A'][:,0,0], '-r')
plt.plot(output2['design'][:,0], output2['A'][:,0,0], '-g')
plt.plot(output1['design'][:,0], output1['A'][:,0,0], '-b')


plt.xlim([0.35, 0.7])
plt.ylim([0.82, 0.87])

design_grid, A11_grid = np.meshgrid(np.linspace(0.35, 0.7, 10),
                                    np.linspace(0.82, 0.87, 10))

plt.tight_layout()
plt.savefig('poisson_vector_field.pdf')

# plt.figure()
# plt.plot(output1['iterations'], output1['logit_design'])
# plt.plot(output2['iterations'], output2['logit_design'])
# plt.plot(output3['iterations'], output3['logit_design'])
# plt.ylim([-0.7, 0.7])
# plt.xlabel('Iterations')
# plt.ylabel('$\lambda$')
# plt.tight_layout()
# plt.savefig('poisson_traceplot_design.pdf')

# plt.figure()
# plt.plot(output1['iterations'], output1['eta11'])
# plt.plot(output2['iterations'], output2['eta11'])
# plt.plot(output3['iterations'], output3['eta11'])
# plt.ylim([-0.205, -0.135])
# plt.xlabel('Iterations')
# plt.ylabel('$a_{11}$')
# plt.tight_layout()
# plt.savefig('poisson_traceplot_param.pdf')

wait = input('Press enter to terminate')
