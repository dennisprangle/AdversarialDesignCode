import numpy as np
import matplotlib.pyplot as plt
import pandas

parameters = {'axes.labelsize':'x-large', 'xtick.labelsize':'x-large',
              'ytick.labelsize':'x-large'}
plt.rcParams.update(parameters)

out_ACE = pandas.read_csv('outputs/pk_SIG.csv', na_values='Inf')
out_ACE = out_ACE.fillna(np.inf)

##################
## EXECUTION TIMES
##################

mean_ace_run_time = np.mean(out_ACE['times'])
print('Mean ACE run time {:.1f}s'.format(mean_ace_run_time))

#####################
## OUTPUT DESIGN PLOTS
#####################

plt.figure()
colnames = ['design_' + str(i+1) for i in range(15)]
ace_designs = out_ACE[colnames].to_numpy()
ace_designs = np.sort(ace_designs, 1)
design_index = [i+1 for i in range(15)]
design_index = np.tile(design_index, ace_designs.shape[0])
plt.scatter(design_index, ace_designs, color='b', alpha=0.05)
plt.ylim([0.,24.])
plt.xlabel('Observation index')
plt.ylabel('Observation time')
plt.tight_layout()
plt.savefig('plots/ACE_designs.pdf')
