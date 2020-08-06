import numpy as np
import matplotlib.pyplot as plt
import pickle

with open('pk_example.pkl', 'rb') as infile:
    out_GDA, out_SGD = pickle.load(infile)

output_GDA_list, final_design_GDA_list, pe_time_GDA_list = zip(*out_GDA)
output_SGD_list, final_design_SGD_list, pe_time_SGD_list = zip(*out_SGD)

##################
## EXECUTION TIMES
##################

gda_run_times = [o['time'][-1] for o in output_GDA_list]
mean_gda_run_time = np.mean(np.array(gda_run_times))
print('Mean GDA run time {:.1f}s'.format(mean_gda_run_time))
sgd_run_times = [o['time'][-1] for o in output_SGD_list]
mean_sgd_run_time = np.mean(np.array(sgd_run_times))
print('Mean SGD run time {:.1f}s'.format(mean_sgd_run_time))

mean_gda_pe_run_time = np.mean(np.array(pe_time_GDA_list))
print('Mean GDA point exchange run time {:.1f}s'.format(mean_gda_pe_run_time))
mean_pe_run_time = np.mean(np.array(pe_time_SGD_list))
print('Mean SGD point exchange run time {:.1f}s'.format(mean_sgd_run_time))

##############
## TRACE PLOTS
##############

output = output_GDA_list[0]

fig_tau = plt.figure()
out_design = output['design']
for i in range(out_design.shape[1]):
  plt.plot(output['iterations'], out_design[:,i])
plt.xlabel('Iterations')
plt.ylabel('Observation time')
plt.tight_layout()
plt.savefig('plots/design_GDA_trace.pdf')

fig_A = plt.figure()
out_A = output['A']
out_A = out_A.reshape((out_A.shape[0], -1))
for i in range(out_A.shape[1]):
  plt.plot(output['iterations'], out_A[:,i])
plt.xlabel('Iterations')
plt.ylabel('A matrix')
plt.tight_layout()
plt.savefig('plots/A_GDA_trace.pdf')

fig_K = plt.figure()
for output in output_GDA_list:
    plt.plot(output['iterations'], output['objectiveK'], color='b', alpha=0.2)
plt.xlabel('Iterations')
plt.ylabel('K objective')
plt.yscale('symlog')
plt.tight_layout()
plt.savefig('plots/K_GDA_trace.pdf')

fig_J = plt.figure()
for output in output_GDA_list:
    plt.plot(output['iterations'], output['objectiveJ'], color='b', alpha=0.2)
plt.xlabel('Iterations')
plt.ylabel('J objective')
plt.tight_layout()
plt.savefig('plots/J_GDA_trace.pdf')

output = output_SGD_list[0]

fig_tau = plt.figure()
out_design = output['design']
for i in range(out_design.shape[1]):
  plt.plot(output['iterations'], out_design[:,i])
plt.xlabel('Iterations')
plt.ylabel('Observation time')
plt.tight_layout()
plt.savefig('plots/design_SGD_trace.pdf')

fig_K = plt.figure()
for output in output_SGD_list:
    plt.plot(output['iterations'], output['objectiveK'], color='b', alpha=0.2)
plt.xlabel('Iterations')
plt.ylabel('K objective')
plt.yscale('symlog')
plt.tight_layout()
plt.savefig('plots/K_SGD_trace.pdf')

fig_J = plt.figure()
for output in output_SGD_list:
    plt.plot(output['iterations'], output['objectiveJ'], color='b', alpha=0.2)
plt.xlabel('Iterations')
plt.ylabel('J objective')
plt.tight_layout()
plt.savefig('plots/J_SGD_trace.pdf')

#####################
## OUTPUT DESIGN PLOTS
#####################

plt.figure()
gda_designs = [np.sort(o['design'][-1,:]) for o in output_GDA_list]
design_index = [i+1 for i in range(15)]
#Commented out lines do a line plot instead
#gda_designs = np.vstack(gda_designs)
#for i in range(100):
#    plt.plot(design_index, gda_designs[i,:], "-")
gda_designs = np.hstack(gda_designs)
design_index = np.tile(design_index, len(output_GDA_list))
plt.scatter(design_index, gda_designs, color='b', alpha=0.05)
plt.xlabel('Observation Index')
plt.ylabel('Time')
plt.tight_layout()
plt.savefig('plots/GDA_designs.pdf')

plt.figure()
pe_designs = np.hstack(final_design_GDA_list)
plt.scatter(design_index, pe_designs, color='b', alpha=0.05)
plt.xlabel('Observation Index')
plt.ylabel('Time')
plt.tight_layout()
plt.savefig('plots/GDA_and_PE_designs.pdf')

plt.figure()
gda_designs = [np.sort(o['design'][-1,:]) for o in output_SGD_list]
design_index = [i+1 for i in range(15)]
#Commented out lines do a line plot instead
#gda_designs = np.vstack(gda_designs)
#for i in range(100):
#    plt.plot(design_index, gda_designs[i,:], "-")
gda_designs = np.hstack(gda_designs)
design_index = np.tile(design_index, len(output_SGD_list))
plt.scatter(design_index, gda_designs, color='b', alpha=0.05)
plt.xlabel('Observation Index')
plt.ylabel('Time')
plt.tight_layout()
plt.savefig('plots/SGD_designs.pdf')

plt.figure()
pe_designs = np.hstack(final_design_SGD_list)
plt.scatter(design_index, pe_designs, color='b', alpha=0.05)
plt.xlabel('Observation Index')
plt.ylabel('Time')
plt.tight_layout()
plt.savefig('plots/SGD_and_PE_designs.pdf')
