## Comparison with a benchmark design used by a reviewer

import numpy as np
import torch
from torch.distributions import Normal, LogNormal, Independent
import matplotlib.pyplot as plt
import pickle
from pk_example import PK_FIM
from pk_SIG import objective

###############################
## SET UP DESIGNS
###############################

with open('outputs/pk_gda_K1.pkl', 'rb') as infile:
    out_GDA = pickle.load(infile)

design_adv = torch.tensor(out_GDA['final_design'][0,:])
design_uni = torch.tensor(np.linspace(1.,15.,15), dtype=torch.float32)

###############################
## DIAGNOSTIC COMPARISONS
###############################

torch.manual_seed(0)
fim = PK_FIM(nsamples=1)
fim.initialise_J_estimation()
diagJ_adv = fim.estimateJ(design_adv.unsqueeze(0)).detach().numpy()[0]
diagJ_uni = fim.estimateJ(design_uni.unsqueeze(0)).detach().numpy()[0]

print("J estimates: adv {:.1e}, unif {:.1e}".format(diagJ_adv, diagJ_uni))

diagSIG_adv = objective(design_adv)
diagSIG_uni = objective(design_uni)

print("SIG estimates: adv {:.1f}, unif {:.1f}".format(diagSIG_adv, diagSIG_uni))

###############################
## POSTERIOR PLOT VS ADV DESIGN
###############################

loc = torch.tensor(np.log((0.1, 1., 20.)))
scale = torch.tensor(np.sqrt((0.05, 0.05, 0.05)))
prior = LogNormal(loc, scale)
prior = Independent(prior, 1)

def get_x(theta, design):
    theta1 = theta[..., 0:1]
    theta2 = theta[..., 1:2]
    theta3 = theta[..., 2:3]
    while design.dim() < theta1.dim():
        design = design.unsqueeze(0)
    x = 400. * theta2 * \
        (torch.exp(-theta1*design) - torch.exp(-theta2*design)) \
        / (theta3*(theta2-theta1))    
    return x

loc = torch.zeros(design_adv.shape)
scale = 0.1 * torch.ones(design_adv.shape)
noise = Normal(loc=loc, scale=scale)
noise = Independent(noise, 1)

torch.manual_seed(0)
theta0 = prior.sample()
noise0 = noise.sample()
x0_adv = get_x(theta0, design_adv)
x0_uni = get_x(theta0, design_uni)
y0_adv = x0_adv + noise0
y0_uni = x0_uni + noise0
print("True parameters", theta0)

loc = theta0
scale = torch.tensor((0.2, 0.2, 0.2))
proposal = Normal(loc, scale)
proposal = Independent(proposal, 1)

def is_weights(theta, design, y0):
    x = get_x(theta, design)
    diff = x - y0.unsqueeze(0)
    log_w = noise.log_prob(diff) - proposal.log_prob(theta)
    log_w -= torch.max(log_w)
    w = torch.exp(log_w)
    w /= torch.sum(w)
    return w

nsamples = 10 ** 6
theta_is = proposal.sample((nsamples,))
w_adv = is_weights(theta_is, design_adv, y0_adv)
w_uni = is_weights(theta_is, design_uni, y0_uni)

xdim = 1
ydim = 2
nplot = 500

plt.figure()

tokeep = (w_adv > 1e-6)
tt = theta_is[tokeep,:]
ww = w_adv[tokeep]
ww = ww.detach().numpy()
ww /= np.sum(ww)
ii = np.random.choice(len(ww), size=nplot, replace=True, p=ww)
post = tt[ii,:]
plt.plot(post[:, xdim], post[:, ydim], "b^", alpha=0.3, markersize=10)

tokeep = (w_uni > 1e-6)
tt = theta_is[tokeep,:]
ww = w_uni[tokeep]
ww = ww.detach().numpy()
ww /= np.sum(ww)
ii = np.random.choice(len(ww), size=nplot, replace=True, p=ww)
post = tt[ii,:]
plt.plot(post[:, xdim], post[:, ydim], "yo", alpha=0.3)

plt.plot(theta0[xdim], theta0[ydim], "kx", markersize=10, markeredgewidth=5)

plt.xlabel(r'$\theta_{:d}$'.format(xdim+1))
plt.ylabel(r'$\theta_{:d}$'.format(ydim+1))
plt.tight_layout()
plt.savefig('plots/PK_posterior_unif_comparison.pdf')
