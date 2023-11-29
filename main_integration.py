
import time
#import scipy.io as sio
import loss_landscape_functions

def get_weight_trajectories(tasks,mask, save_name):
    
    iterations = 1000
    steps = 100
    #mask = np.ones((100,100))
    randomize_task_order = 1
    lr = 0.001
    #hidden_size = 100
    
    #sample_weights
    for iterr in range(iterations):
        print("iterations: ", iterr)
        start_time = time.time()
        
        net, loss_trajectory, weight_traj = loss_landscape_functions.train_simultaneous_integration(tasks,steps,mask,lr)
        if iterr == 0:
            total_weight_traj = weight_traj
        else:
            total_weight_traj = np.append(total_weight_traj, weight_traj, axis=0)
        
        if (iterr+1)%500 == 0:
            np.savez_compressed(save_name,total_weight_traj = total_weight_traj)
        
        print("time: ",time.time() - start_time)
        #mdic = {"total_weight_traj": total_weight_traj}

        #sio.savemat("total_weight_traj.mat", mdic)

    return total_weight_traj

import numpy as np

tasks = ["GoNogo-v0","PerceptualDecisionMaking-v0"]
mask = np.zeros((100,100))
save_name = "integration_task.npz"

total_weight_traj = get_weight_trajectories(tasks,mask, save_name)

np.savez_compressed(save_name,total_weight_traj = total_weight_traj)