import matplotlib.pyplot as plt
import pickle
plt.ion()

with open('pk_example.pkl', 'rb') as infile:
    out = pickle.load(infile)

output_list, final_design_list, point_exchange_time_list = zip(*out)

output = output_list[0]

plt.figure()
out_design = output['design']
for i in range(out_design.shape[1]):
  plt.plot(output['iterations'], out_design[:,i])

plt.xlabel('Iterations')
plt.ylabel('Observation time')
plt.tight_layout()

plt.figure()
out_A = output['A']
out_A = out_A.reshape((out_A.shape[0], -1))
for i in range(out_A.shape[1]):
  plt.plot(output['iterations'], out_A[:,i])

plt.xlabel('Iterations')
plt.ylabel('A matrix')
plt.tight_layout()

plt.figure()
plt.plot(output['iterations'], output['objectiveK'])
plt.xlabel('Iterations')
plt.ylabel('K objective')
plt.yscale('symlog')
plt.tight_layout()

plt.figure()
for output in output_list:
    plt.plot(output['iterations'], output['objectiveJ'])
plt.xlabel('Iterations')
plt.ylabel('J objective')
plt.tight_layout()

wait = input('Press enter to terminate')

## TO DO: SAVE OUTPUTS (TO A FOLDER), USE SOPHIE'S CODE FOR J OBJECTIVE PLOT APPEARANCE
