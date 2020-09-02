import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle
import advOpt
from poisson_example import Poisson_FIM

##############
##VECTOR FIELD
##############

fim = Poisson_FIM(omega=(2., 1.))

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
    objective = fim.estimateK(design.unsqueeze(0), A.unsqueeze(0))
    objective.backward()
    logit_design_grad_grid[i,j] = float(logit_design.grad)
    eta11_grad_grid[i,j] = float(eta11.grad)

vector_field_plot = plt.quiver(logit_design_grid, eta11_grid,
                               -logit_design_grad_grid, eta11_grad_grid/1000,
                               angles='xy')

vector_field_plot.axes.set_xlabel(r'$\lambda$')
vector_field_plot.axes.set_ylabel(r'$\eta_{11}$')

##############
##OPTIMISATION OUTPUT
##############

with open('outputs/poisson3.pkl', 'rb') as infile:
    output3 = pickle.load(infile)

with open('outputs/poisson4.pkl', 'rb') as infile:
    output4 = pickle.load(infile)

with open('outputs/poisson5.pkl', 'rb') as infile:
    output5 = pickle.load(infile)

logit = lambda x: np.log(x/(1-x))
dot_indices = range(99, 5000, 100)
toplot = [(output3, 'b'), (output4, 'g'), (output5, 'r')]
for (o, col) in toplot:
  plt.plot(logit(o['design'][dot_indices,0,0]),
           np.log(o['A'][dot_indices,0,0,0]), 'o' + col)
  plt.plot(logit(o['design'][:,0,0]), np.log(o['A'][:,0,0,0]), '-' + col)

plt.xlim([-0.7, 0.7])
plt.ylim([-0.203, -0.137])

plt.tight_layout()
plt.savefig('plots/poisson_vector_field.pdf')

plt.figure()
for (o, _) in toplot:
  plt.plot(o['iterations'], logit(o['design'][:,0,0]))
plt.ylim([-0.7, 0.7])
plt.xlabel('Iterations')
plt.ylabel(r'$\lambda$')
plt.tight_layout()
plt.savefig('plots/poisson_traceplot_design.pdf')

plt.figure()
for (o, _) in toplot:
  plt.plot(o['iterations'], np.log(o['A'][:,0,0,0]))
plt.ylim([-0.205, -0.135])
plt.xlabel('Iterations')
plt.ylabel(r'$\eta_{11}$')
plt.tight_layout()
plt.savefig('plots/poisson_traceplot_param.pdf')
