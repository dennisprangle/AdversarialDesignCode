import numpy as np
import matplotlib.pyplot as plt
import pickle

parameters = {'axes.labelsize':'x-large', 'xtick.labelsize':'x-large',
              'ytick.labelsize':'x-large'}
plt.rcParams.update(parameters)

with open('outputs/pk_gda_K100.pkl', 'rb') as infile:
    out_GDA_K100 = pickle.load(infile)

with open('outputs/pk_gda_K10.pkl', 'rb') as infile:
    out_GDA_K10 = pickle.load(infile)

with open('outputs/pk_gda_K1.pkl', 'rb') as infile:
    out_GDA_K1 = pickle.load(infile)

with open('outputs/pk_sgd.pkl', 'rb') as infile:
    out_SGD = pickle.load(infile)

##################
## EXECUTION TIMES
##################

mean_gda_run_time = out_GDA_K1['time'][-1]/100.
print('Mean GDA run time {:.1f}s'.format(mean_gda_run_time))
mean_sgd_run_time = out_SGD['time'][-1]/100.
print('Mean SGD run time {:.1f}s'.format(mean_sgd_run_time))

mean_gda_pe_run_time = out_GDA_K1['point_exchange_time'] / 100.
print('Mean GDA point exchange run time {:.1f}s'.format(mean_gda_pe_run_time))
mean_sgd_pe_run_time = out_SGD['point_exchange_time'] / 100.
print('Mean SGD point exchange run time {:.1f}s'.format(mean_sgd_pe_run_time))

##############
## TRACE PLOTS
##############

plt.figure()
gda_design0 = out_GDA_K1['design'][:,0,:]
gda_its = out_GDA_K1['iterations']
for i in range(gda_design0.shape[1]):
  plt.plot(gda_its, gda_design0[:,i])
plt.xlabel('Iterations')
plt.ylabel('Observation time')
plt.tight_layout()
plt.savefig('plots/design_GDA_trace.pdf')

plt.figure()
gda_A0 = out_GDA_K1['A'][:,0,:,:]
gda_A0 = gda_A0.reshape((gda_A0.shape[0], -1))
for i in range(gda_A0.shape[1]):
  plt.plot(gda_its, gda_A0[:,i])
plt.xlabel('Iterations')
plt.ylabel('A matrix')
plt.tight_layout()
plt.savefig('plots/A_GDA_trace.pdf')

plt.figure()
gda_K = out_GDA_K1['objectiveK']
gda_its = out_GDA_K1['iterations']
for i in range(gda_K.shape[1]):
    plt.plot(gda_its, gda_K[:,i], color='b', alpha=0.2)
plt.xlabel('Iterations')
plt.ylabel('K objective')
plt.yscale('symlog')
plt.tight_layout()
plt.savefig('plots/K_GDA_trace.pdf')

plt.figure()
gda_J = out_GDA_K1['objectiveJ']
for i in range(gda_J.shape[1]):
    plt.plot(gda_its, gda_J[:,i], color='b', alpha=0.2)
plt.xlabel('Iterations')
plt.ylabel('J objective')
plt.tight_layout()
plt.savefig('plots/J_GDA_trace.pdf')

plt.figure()
sgd_design0 = out_SGD['design'][:,0,:]
sgd_its = out_SGD['iterations']
for i in range(sgd_design0.shape[1]):
  plt.plot(sgd_its, sgd_design0[:,i])
plt.xlabel('Iterations')
plt.ylabel('Observation time')
plt.tight_layout()
plt.savefig('plots/design_SGD_trace.pdf')

plt.figure()
sgd_K = out_SGD['objectiveK']
sgd_its = out_SGD['iterations']
for i in range(sgd_K.shape[1]):
    plt.plot(sgd_its, sgd_K[:,i], color='b', alpha=0.2)
plt.xlabel('Iterations')
plt.ylabel('K objective')
plt.yscale('symlog')
plt.tight_layout()
plt.savefig('plots/K_SGD_trace.pdf')

fig_J_sgd = plt.figure()
sgd_J = out_SGD['objectiveJ']
for i in range(sgd_K.shape[1]):
    plt.plot(sgd_its, sgd_J[:,i], color='b', alpha=0.2)
plt.xlabel('Iterations')
plt.ylabel('J objective')
plt.tight_layout()
plt.savefig('plots/J_SGD_trace.pdf')

#####################
## TRACE PLOTS FOR VARYING K
#####################

plt.figure()
gda_J_100 = out_GDA_K100['objectiveJ']
gda_J_10 = out_GDA_K10['objectiveJ']
gda_J_1 = out_GDA_K1['objectiveJ']
gda_times_K100 = out_GDA_K100['time']
gda_times_K10 = out_GDA_K10['time']
gda_times_K1 = out_GDA_K1['time']
for i in range(gda_J_100.shape[1]):
    plt.plot(gda_times_K100, gda_J_100[:,i], color='b', alpha=0.2)
    plt.plot(gda_times_K10, gda_J_10[:,i], color='r', alpha=0.2)
    plt.plot(gda_times_K1, gda_J_1[:,i], color='g', alpha=0.2)

plt.xlim([0,300])
plt.xlabel('Runtime (seconds)')
plt.ylabel('J objective')
plt.tight_layout()
plt.savefig('plots/J_GDA_comparison_time.pdf')

plt.figure()
gda_its_100 = out_GDA_K100['iterations']
gda_its_10 = out_GDA_K10['iterations']
gda_its_1 = out_GDA_K1['iterations']
for i in range(gda_J_100.shape[1]):
    plt.plot(gda_its_100, gda_J_100[:,i], color='b', alpha=0.2)
    plt.plot(gda_its_10, gda_J_10[:,i], color='r', alpha=0.2)
    plt.plot(gda_its_1, gda_J_1[:,i], color='g', alpha=0.2)

plt.xlim([0,100000])
plt.xlabel('Iterations')
plt.ylabel('J objective')
plt.tight_layout()
plt.savefig('plots/J_GDA_comparison_iterations.pdf')

#####################
## OUTPUT DESIGN PLOTS
#####################

plt.figure()
gda_designs = out_GDA_K1['design'][-1,...]
gda_designs = np.sort(gda_designs, 1)
design_index = [i+1 for i in range(15)]
#Commented out lines do a line plot instead
#gda_designs = np.vstack(gda_designs)
#for i in range(100):
#    plt.plot(design_index, gda_designs[i,:], "-")
design_index = np.tile(design_index, gda_designs.shape[0])
gda_designs = np.hstack(gda_designs)
plt.scatter(design_index, gda_designs, color='b', alpha=0.05)
plt.ylim([0.,24.])
plt.xlabel('Observation index')
plt.ylabel('Observation time')
plt.tight_layout()
plt.savefig('plots/GDA_designs.pdf')

plt.figure()
gda_pe_designs = np.hstack(out_GDA_K100['final_design'])
plt.scatter(design_index, gda_pe_designs, color='b', alpha=0.05)
plt.ylim([0.,24.])
plt.xlabel('Observation index')
plt.ylabel('Observation time')
plt.tight_layout()
plt.savefig('plots/GDA_and_gda_pe_designs.pdf')

plt.figure()
sgd_designs = out_SGD['design'][-1,...]
sgd_designs = np.sort(sgd_designs, 1)
design_index = [i+1 for i in range(15)]
#Commented out lines do a line plot instead
#sgd_designs = np.vstack(sgd_designs)
#for i in range(100):
#    plt.plot(design_index, sgd_designs[i,:], "-")
design_index = np.tile(design_index, sgd_designs.shape[0])
sgd_designs = np.hstack(sgd_designs)
plt.scatter(design_index, sgd_designs, color='b', alpha=0.05)
plt.ylim([0.,24.])
plt.xlabel('Observation index')
plt.ylabel('Observation time')
plt.tight_layout()
plt.savefig('plots/SGD_designs.pdf')

plt.figure()
sgd_pe_designs = np.hstack(out_SGD['final_design'])
plt.scatter(design_index, sgd_pe_designs, color='b', alpha=0.05)
plt.ylim([0.,24.])
plt.xlabel('Observation index')
plt.ylabel('Observation time')
plt.tight_layout()
plt.savefig('plots/SGD_and_PE_designs.pdf')
