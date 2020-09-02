import numpy as np
import matplotlib.pyplot as plt
import pickle

scales = [0.01, 0.02, 0.04, 0.08]
outputs = []

for i in range(4):
    filename = './outputs/geo' + str(i+1) + '.pkl'
    with open(filename, 'rb') as infile:
        outputs.append(pickle.load(infile))

##################
## EXECUTION TIMES
##################

run_times = [o['time'][-1] for o in outputs]
mean_run_time = np.mean(np.array(run_times))
print('Mean run time {:.1f}s'.format(mean_run_time))
    
########
##PLOTS
########

for i in range(len(scales)):
    s = scales[i]
    o = outputs[i]
    out_design = o['design'].squeeze(1)

    plt.figure()
    for j in range(out_design.shape[1]):
        plt.plot(out_design[:,j,0], out_design[:,j,1], '-')
    plt.plot(out_design[-1,:,0], out_design[-1,:,1], 'o')
    plt.tight_layout()
    plt.savefig('plots/geo_design_trace_{:d}.pdf'.format(i))
    
    plt.figure()
    plt.plot(out_design[-1,:,0], out_design[-1,:,1], 'o')
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig('plots/geo_design_{:d}.pdf'.format(i))

    plt.figure()
    out_A = o['A']
    out_A = out_A.reshape((out_A.shape[0], -1))
    for j in range(out_A.shape[1]):
        plt.plot(o['iterations'], out_A[:,j])
    plt.xlabel('Iterations')
    plt.ylabel('A matrix')
    plt.tight_layout()
    plt.savefig('plots/geo_A_trace_{:d}.pdf'.format(i))

    plt.figure()
    plt.plot(o['iterations'], o['objectiveK'])
    plt.xlabel('Iterations')
    plt.ylabel('K objective')
    plt.yscale('symlog')
    plt.tight_layout()
    plt.savefig('plots/geo_K_objective_{:d}.pdf'.format(i))

    plt.figure()
    plt.plot(o['iterations'], o['objectiveJ'])
    plt.xlabel('Iterations')
    plt.ylabel('J objective')
    plt.title(r'$\ell=${:.2f}'.format(s))
    plt.tight_layout()
    plt.savefig('plots/geo_J_objective_{:d}.pdf'.format(i))
