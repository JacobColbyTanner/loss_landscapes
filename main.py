

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

from sklearn.decomposition import PCA
import time
#import scipy.io as sio

def get_weight_trajectories(tasks,mask, save_name):
    
    iterations = 700
    steps = 100
    #mask = np.ones((100,100))
    randomize_task_order = 1
    lr = 0.001
    #hidden_size = 100
    
    #sample_weights
    for iterr in range(iterations):
        print("iterations: ", iterr)
        start_time = time.time()
        net, loss_traj, weight_traj = loss_landscape_functions.train_multitask2(tasks,steps,mask,lr,randomize_task_order)

        if iterr == 0:
            total_weight_traj = weight_traj
        else:
            total_weight_traj = np.append(total_weight_traj, weight_traj, axis=0)
        
        #if (iterr+1)%100 == 0:
            #np.savez_compressed(save_name,total_weight_traj = total_weight_traj)
        
        print("time: ",time.time() - start_time)
        #mdic = {"total_weight_traj": total_weight_traj}

        #sio.savemat("total_weight_traj.mat", mdic)

    return total_weight_traj



mask = np.ones((80,80))

save_name = "/N/project/networkRNNs/loss_lndscape_data/four_tasks_no_modules.npz"

total_weight_traj = get_weight_trajectories(tasks,mask,save_name)

np.savez_compressed(save_name,total_weight_traj = total_weight_traj)


