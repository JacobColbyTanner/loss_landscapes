

import numpy
import loss_landscape_functions
import neurogym 

from gym.envs.registration import register

register(id = 'gonogo_variable_delay-v0',entry_point= "neurogym.envs.gonogo_variable_delay:gonogo_variable_delay")

import gym

env = gym.make('gonogo_variable_delay-v0')  # Replace 'YourEnvName-v0' with the ID you used during registration


from gym.envs.registration import register

register(id = 'OrientedBar-v0',entry_point= "neurogym.envs.OrientedBar7:OrientedBar7")



env = gym.make('OrientedBar-v0')  # Replace 'YourEnvName-v0' with the ID you used during registration




from gym.envs.registration import register

register(id = 'VisMotorReaching-v0',entry_point= "neurogym.envs.VisMotorReaching19:VisMotorReaching19")



env = gym.make('VisMotorReaching-v0')  # Replace 'YourEnvName-v0' with the ID you used during registration



from gym.envs.registration import register

register(id = 'ObjectSequenceMemory-v0',entry_point= "neurogym.envs.ObjectSequenceMemory24:ObjectSequenceMemory24")



env = gym.make('ObjectSequenceMemory-v0')  # Replace 'YourEnvName-v0' with the ID you used during registration



tasks = ['PerceptualDecisionMaking-v0','gonogo_variable_delay-v0','ObjectSequenceMemory-v0','OrientedBar-v0'] #'GoNogo-v0',



import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

 #create modular pruning mask with stochastic block model
sizes = [25,25,25,25]
off_block = 0.1
probs = [[1, off_block,off_block,off_block],[off_block, 1, off_block, off_block],[off_block, off_block, 1, off_block],[off_block, off_block, off_block, 1]]
g = nx.stochastic_block_model(sizes,probs,directed=True)
gg = nx.to_numpy_array(g)
I = np.eye(gg.shape[0])
gg = gg+I

plt.imshow(gg)
plt.show()

mask = gg

save_name = "/N/project/networkRNNs/loss_lndscape_data/four_tasks_all_weights_modules.npz"

total_weight_traj = loss_landscape_functions.get_weight_trajectories(tasks,mask,save_name)

np.savez_compressed(save_name,total_weight_traj = total_weight_traj)


