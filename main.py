

import numpy
import loss_landscape_functions


tasks = ['PerceptualDecisionMaking-v0','gonogo_variable_delay-v0','ObjectSequenceMemory-v0','OrientedBar-v0'] #'GoNogo-v0',
mask = np.ones((100,100))

total_weight_traj = get_weight_trajectories(tasks,mask)

np.savez_compressed("four_tasks_all_weights.npz",total_weight_traj = total_weight_traj)





import numpy as np

TWJ = np.load("four_tasks_all_weights.npz")

data = TWJ["total_weight_traj"]



ndim = 3
max_corr = False

#data = TWJ["total_weight_traj"]
all_corr = pca_validate(data,ndim,max_corr)



import matplotlib.pyplot as plt


plt.plot(np.absolute(all_corr))
plt.xlabel("number of iterations")
plt.ylabel("similarity to other sample (N=500)")
plt.title("Perceptual Decision-Making & Go vs. No-Go")
plt.savefig("Perceptual_Decision_Making_Go_vs_No-Go.png", dpi=300, format='png', bbox_inches='tight')
plt.show()
