import time
#import scipy.io as sio
import loss_landscape_functions
import matplotlib.pyplot as plt
import numpy as np

num_iter = 10
steps = 1000
lr = 0.001
tasks = ["GoNogo-v0","PerceptualDecisionMaking-v0"]
mask_RC = np.load("data/RC_network.npy")
mask_no_RC = np.load("data/no_RC_network.npy")

loss_RC = []
loss_no_RC = []

for iter in range(num_iter):    
    print("iter: ", iter)
    print("RC:")
    net, loss_trajectory_RC, _ = loss_landscape_functions.train_simultaneous_integration(tasks,steps,mask_RC,lr)

    loss_RC.append(loss_trajectory_RC)
    print("no RC:")
    net, loss_trajectory_no_RC, _ = loss_landscape_functions.train_simultaneous_integration(tasks,steps,mask_no_RC,lr)

    loss_no_RC.append(loss_trajectory_no_RC)



np.save("data/loss_RC.npy",loss_RC)

np.save("data/loss_no_RC.npy",loss_no_RC)

plt.figure()
plt.plot(np.mean(loss_RC,axis=0))
plt.plot(np.mean(loss_no_RC,axis=0))
plt.title("Loss: RC vs no RC")
plt.savefig("figures/loss_traj_compare.png")