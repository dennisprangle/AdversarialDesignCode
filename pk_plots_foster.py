import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pickle

parameters = {'axes.labelsize':'x-large', 'xtick.labelsize':'x-large',
              'ytick.labelsize':'x-large'}
plt.rcParams.update(parameters)

with open('outputs/pk-pce-1hour.pickle', 'rb') as infile:
    out_PCE_1hour = pickle.load(infile)

with open('outputs/pk-pce-2hours.pickle', 'rb') as infile:
    out_PCE_2hours = pickle.load(infile)

with open('outputs/pk-ba.pickle', 'rb') as infile:
    out_BA = pickle.load(infile)

##################
## EXECUTION TIMES
##################

mean_run_time = out_PCE_1hour['wall_times'][-1]/100.
print('Mean run time PCE short {:.1f}s'.format(mean_run_time))

mean_run_time = out_BA['wall_times'][-1]/100.
print('Mean run time BA {:.1f}s'.format(mean_run_time))

############################
## TRACE PLOT OF BA RUNS
############################

designs_BA = out_BA['xi_history'].detach().numpy()

#First replication
plt.figure()
times_BA = out_BA['wall_times']
for i in range(15):
   plt.plot(times_BA[:1700], designs_BA[:1700,0,i], "-")
plt.xlabel('Runtime (seconds)')
plt.ylabel('Observation time')
plt.tight_layout()
plt.savefig('plots/BA_trace.pdf')

#Several replications
col_list = [x for x in mcolors.TABLEAU_COLORS.values()]
plt.figure()
times_BA = out_BA['wall_times']
for rep in range(10):
    for i in range(15):
        plt.plot(times_BA[1:], designs_BA[1:,rep,i], "-", color=col_list[rep])
plt.xlabel('Runtime')
plt.ylabel('Observation time')
plt.tight_layout()

######################
## OUTPUT DESIGN PLOTS
######################

designs1 = out_PCE_1hour['xi_history'].detach().numpy()
designs2 = out_PCE_2hours['xi_history'].detach().numpy()
designs_BA = out_BA['xi_history'].detach().numpy()

plt.figure()
final_designs = np.sort(designs1[-1,:,:], 1)
design_index = [i+1 for i in range(15)]
design_index = np.tile(design_index, final_designs.shape[0])
stacked_designs = np.hstack(final_designs)
plt.scatter(design_index, stacked_designs, color='b', alpha=0.05)
plt.ylim([0.,24.])
plt.xlabel('Observation index')
plt.ylabel('Observation time')
plt.tight_layout()
plt.savefig('plots/PCE_designs1.pdf')

plt.figure()
final_designs = np.sort(designs2[-1,:,:], 1)
design_index = [i+1 for i in range(15)]
design_index = np.tile(design_index, final_designs.shape[0])
stacked_designs = np.hstack(final_designs)
plt.scatter(design_index, stacked_designs, color='b', alpha=0.05)
plt.ylim([0.,24.])
plt.xlabel('Observation index')
plt.ylabel('Observation time')
plt.tight_layout()
plt.savefig('plots/PCE_designs2.pdf')

plt.figure()
final_designs = np.sort(designs_BA[-1,:,:], 1)
design_index = [i+1 for i in range(15)]
design_index = np.tile(design_index, final_designs.shape[0])
stacked_designs = np.hstack(final_designs)
plt.scatter(design_index, stacked_designs, color='b', alpha=0.05)
plt.ylim([0.,24.])
plt.xlabel('Observation index')
plt.ylabel('Observation time')
plt.tight_layout()
plt.savefig('plots/BA_designs.pdf')
