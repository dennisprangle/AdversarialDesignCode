import numpy as np
import torch
from torch.distributions import Normal, LogNormal, Independent
import matplotlib.pyplot as plt
import pickle

################
## IMPORT DESIGN
################

with open('outputs/pk_gda.pkl', 'rb') as infile:
    out_GDA = pickle.load(infile)

with open('outputs/pk_sgd.pkl', 'rb') as infile:
    out_SGD = pickle.load(infile)

design_adv = torch.tensor(out_GDA['final_design'][0,:])
design_fig = torch.tensor(out_SGD['final_design'][0,:])

#######################
## PRIOR AND MODEL
#######################

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

#######################
## GENERATE PSEUDO-DATA
#######################

torch.manual_seed(0)
theta0 = prior.sample()
noise0 = noise.sample()
x0_adv = get_x(theta0, design_adv)
x0_fig = get_x(theta0, design_fig)
y0_adv = x0_adv + noise0
y0_fig = x0_fig + noise0

######################
## IMPORTANCE SAMPLING
######################

def is_weights(theta, design, y0):
    x = get_x(theta, design)
    diff = x - y0.unsqueeze(0)
    log_w = noise.log_prob(diff) - prior.log_prob(theta)
    log_w -= torch.max(log_w)
    w = torch.exp(log_w)
    w /= torch.sum(w)
    return w

nsamples = 10 ** 7
theta_is = prior.sample((nsamples,))
w_adv = is_weights(theta_is, design_adv, y0_adv)
w_fig = is_weights(theta_is, design_fig, y0_fig)

#######
## PLOT
#######

xdim = 0
ydim = 2
nplot = 500

plt.figure()

plt.plot(theta_is[0:5000, xdim], theta_is[0:5000, ydim], "go", alpha=0.3)

tokeep = (w_adv > 1e-6)
tt = theta_is[tokeep,:]
ww = w_adv[tokeep]
ww = ww.detach().numpy()
ww /= np.sum(ww)
ii = np.random.choice(len(ww), size=nplot, replace=True, p=ww)
post = tt[ii,:]
plt.plot(post[:, xdim], post[:, ydim], "b+", alpha=0.5)

tokeep = (w_fig > 1e-6)
tt = theta_is[tokeep,:]
ww = w_fig[tokeep]
ww = ww.detach().numpy()
ww /= np.sum(ww)
ii = np.random.choice(len(ww), size=nplot, replace=True, p=ww)
post = tt[ii,:]
plt.plot(post[:, xdim], post[:, ydim], "r^", alpha=0.5)

plt.plot(theta0[xdim], theta0[ydim], "kx", markersize=10)

plt.xlabel(r'$\theta_{:d}$'.format(xdim+1))
plt.ylabel(r'$\theta_{:d}$'.format(ydim+1))
plt.tight_layout()
plt.savefig('plots/PK_posterior.pdf')
