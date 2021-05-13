import numpy as np
import matplotlib.pyplot as plt
import pandas
import torch
from torch.distributions import Normal
import pk_example

out_ACE = pandas.read_csv('outputs/pk_SIG_tuning.csv', na_values='Inf')
out_ACE = out_ACE.fillna(np.inf)

##################
## EXECUTION TIMES
##################

time_pairs = out_ACE.groupby(['mc_size_gp'])['times']
mc_size, times = [x for x in zip(*time_pairs)]
plt.boxplot(times, whis=(0,100))
locs, _ = plt.xticks()
plt.xlabel("Monte Carlo sample size")
plt.ylabel("Run time (seconds)")
plt.xticks(ticks=locs, labels=mc_size)
plt.tight_layout()
plt.savefig('plots/pk_tuning_times.pdf')

#####################
## OUTPUT DESIGN PLOTS
#####################

plt.ion()
colnames = ['design_' + str(i+1) for i in range(15)]
ace_grouped = out_ACE.groupby(['mc_size_gp'])

for pars, dat in ace_grouped:
    gp = pars
    plt.figure()
    ace_designs = dat[colnames].to_numpy()
    ace_designs = np.sort(ace_designs, 1)
    design_index = [i+1 for i in range(15)]
    design_index = np.tile(design_index, ace_designs.shape[0])
    plt.scatter(design_index, ace_designs, color='b', alpha=0.05)
    plt.ylim([0.,24.])
    plt.xlabel('Observation index')
    plt.ylabel('Observation time')
    plt.title('GP {}'.format(gp))
    plt.tight_layout()

################################
## CODE TO ESTIMATE EXPECTED SIG
################################

fim = pk_example.PK_FIM(nsamples=100) 

def pk_mean(theta, design):
  design = design.unsqueeze(1)
  theta1 = theta[:, 0:1].unsqueeze(0)
  theta2 = theta[:, 1:2].unsqueeze(0)
  theta3 = theta[:, 2:3].unsqueeze(0)
  x = 400. * theta2 * (torch.exp(-theta1*design) - torch.exp(-theta2*design)) / (theta3*(theta2-theta1))
  return x

def estimate_SIG(design, n_outer=1000, n_inner=1000):
  torch.manual_seed(0) ## Should reduce variability
  
  if len(design.shape) == 1:
    design = design.unsqueeze(0)
  n_designs = design.shape[0]
  
  noise_dist = Normal(loc=0., scale=0.1)
  thetas_outer = fim.prior.sample((n_outer,))
  thetas_inner = fim.prior.sample((n_inner,))
  y_outer = pk_mean(thetas_outer, design) + noise_dist.sample((1, n_outer, 15))
  ## Same noise reused for all designs to reduce variability
  x_inner = pk_mean(thetas_inner, design)
  ##y_outer and x_inner dimensions represent counts of:
  ##design, x or y replication, observation

  ## A simple but inefficient approach is:
  ## Outer loop over designs
    ## Middle loop over ys
      ## Inner loop over thetas
        ## Evaluate likelihood f(y | theta; design)
      ## Take mean of likelihoods to get evidence estimate
    ## Get SIG estimate: entropy minus mean of log evidences
  ## Return vector of SIG estimates
  
  ## Whole calculation could be parallelised, but I don't have enough memory
  ## So I iterate over designs
  def est_SIG(design_count):
    temp = y_outer[design_count,:,:].unsqueeze(1) \
           - x_inner[design_count,:,:].unsqueeze(0)
    ## temp[i,j,k] is y_outer[design_count,i,k] - x_inner[design_count,j,k]

    temp = noise_dist.log_prob(temp)
    ## temp[i,j,k] is log density of observing y_outer[design_count,i,k]
    ## given x_inner[design_count,j,k]

    temp = torch.sum(temp, dim=2)
    ## temp[i,j] is log density of observing y_outer[design_count,i,:]
    ## given thetas_inner[j,:]
  
    temp = torch.logsumexp(temp, dim=1) - np.log(n_outer)
    ## temp[i] is log mean density of observing y_outer[design_count,i,:]
    ## i.e. log evidence estimate

    return 15. * noise_dist.entropy() - torch.mean(temp)
    ## SIG estimate

  return np.array([est_SIG(i).item() for i in range(n_designs)])

################################
## ESTIMATE EXPECTED SIG
################################
sig = []
labels = []
for pars, dat in ace_grouped:
    gp = pars
    ace_designs = dat[colnames].to_numpy()
    ace_designs = torch.tensor(ace_designs)
    ace_SIG = estimate_SIG(ace_designs)
    next_label = 'gp {}'.format(gp)
    labels += [next_label]
    sig += [ace_SIG]

plt.figure()
plt.boxplot(sig, whis=(0,100))
locs, _ = plt.xticks()
plt.ylabel("SIG objective estimate")
plt.xticks(ticks=locs, labels=labels)
plt.tight_layout()
plt.savefig('plots/pk_tuning_SIGs.pdf')
