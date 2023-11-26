#!/N/soft/sles15/deeplearning/Python-3.10.10/python

'''
import subprocess

# Load the module from within the Python script
subprocess.run(['module', 'load', 'python/gpu/3.10.10'], shell=True)
subprocess.run(['module', 'load', 'python'], shell=True)
'''

# Define networks
import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F
import math
    

class CTRNN(nn.Module):
    """Continuous-time RNN.

    Parameters:
        input_size: Number of input neurons
        hidden_size: Number of hidden neurons
        dt: discretization time step in ms. 
            If None, dt equals time constant tau

    Inputs:
        input: tensor of shape (seq_len, batch, input_size)
        hidden: tensor of shape (batch, hidden_size), initial hidden activity
            if None, hidden is initialized through self.init_hidden()
        
    Outputs:
        output: tensor of shape (seq_len, batch, hidden_size)
        hidden: tensor of shape (batch, hidden_size), final hidden activity
    """

    def __init__(self, input_size, hidden_size, dt=None, **kwargs):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.tau = 100
        if dt is None:
            alpha = 1
        else:
            alpha = dt / self.tau
        self.alpha = alpha

        self.input2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)

    def init_hidden(self, input_shape):
        batch_size = input_shape[1]
        return torch.zeros(batch_size, self.hidden_size)

    def recurrence(self, input, hidden):
        """Run network for one time step.
        
        Inputs:
            input: tensor of shape (batch, input_size)
            hidden: tensor of shape (batch, hidden_size)
        
        Outputs:
            h_new: tensor of shape (batch, hidden_size),
                network activity at the next time step
        """
        h_new = torch.relu(self.input2h(input) + self.h2h(hidden))
        h_new = hidden * (1 - self.alpha) + h_new * self.alpha
        return h_new

    def forward(self, input, hidden=None):
        """Propogate input through the network."""
        
        # If hidden activity is not provided, initialize it
        if hidden is None:
            hidden = self.init_hidden(input.shape).to(input.device)

        # Loop through time
        output = []
        input_projection = []
        steps = range(input.size(0))
        for i in steps:
            hidden = self.recurrence(input[i], hidden)
            output.append(hidden)
            input_projection.append(self.input2h(input[i]))

        # Stack together output from all time steps
        output = torch.stack(output, dim=0)  # (seq_len, batch, hidden_size)
        input_projection = torch.stack(input_projection, dim=0)  
        
        return output, hidden, input_projection


class RNNNet(nn.Module):
    """Recurrent network model.

    Parameters:
        input_size: int, input size
        hidden_size: int, hidden size
        output_size: int, output size
    
    Inputs:
        x: tensor of shape (Seq Len, Batch, Input size)

    Outputs:
        out: tensor of shape (Seq Len, Batch, Output size)
        rnn_output: tensor of shape (Seq Len, Batch, Hidden size)
    """
    def __init__(self, input_size, hidden_size, output_size, **kwargs):
        super().__init__()

        # Continuous time RNN
        self.rnn = CTRNN(input_size, hidden_size, **kwargs)
        
        # Add an output layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        rnn_output, hidden, input_projection = self.rnn(x)
        out = self.fc(rnn_output)
        return out, rnn_output
    
    
    










# Define networks
import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F
import math
import neurogym as ngym
import numpy as np
import torch.nn.utils.prune as prune
import gym  # package for RL environments
import torch.optim as optim
import time
import matplotlib.pyplot as plt



def weight_shape_to_flat(net):
    weight_trajectory = net.rnn.input2h.weight.cpu().detach().numpy()
    weight_trajectory = np.append(weight_trajectory,net.rnn.input2h.bias.cpu().detach().numpy())
    weight_trajectory = np.append(weight_trajectory,net.rnn.h2h.weight_orig.cpu().detach().numpy())
    weight_trajectory = np.append(weight_trajectory,net.rnn.h2h.bias.cpu().detach().numpy())
    weight_trajectory = np.append(weight_trajectory,net.fc.weight.cpu().detach().numpy())
    weight_trajectory = np.append(weight_trajectory,net.fc.bias.cpu().detach().numpy())
    
    return weight_trajectory



def weight_flat_to_shape(net,weight):
    weight_shapes = []
    weight_shapes.append(net.rnn.input2h.weight.cpu().detach().numpy().shape)
    weight_shapes.append(net.rnn.input2h.bias.cpu().detach().numpy().shape)
    weight_shapes.append(net.rnn.h2h.weight_orig.cpu().detach().numpy().shape)
    weight_shapes.append(net.rnn.h2h.bias.cpu().detach().numpy().shape)
    weight_shapes.append(net.fc.weight.cpu().detach().numpy().shape)
    weight_shapes.append(net.fc.bias.cpu().detach().numpy().shape)
    
    all_weights = []
    all_weights.append(net.rnn.input2h.weight.cpu().detach().numpy())
    all_weights.append(net.rnn.input2h.bias.cpu().detach().numpy())
    all_weights.append(net.rnn.h2h.weight_orig.cpu().detach().numpy())
    all_weights.append(net.rnn.h2h.bias.cpu().detach().numpy())
    all_weights.append(net.fc.weight.cpu().detach().numpy())
    all_weights.append(net.fc.bias.cpu().detach().numpy())
    
    
    total = np.prod(weight_shapes[0])
    start = 0
    stop = total
    net.rnn.input2h.weight = nn.Parameter(torch.from_numpy(np.reshape(weight[start:stop],weight_shapes[0])).float())
    
    
    
    start = start+total
    total = np.prod(weight_shapes[1])
    stop = stop+total
    net.rnn.input2h.bias = nn.Parameter(torch.from_numpy(np.reshape(weight[start:stop],weight_shapes[1])).float())
    
    start = start+total
    total = np.prod(weight_shapes[2])
    stop = stop+total
    net.rnn.h2h.weight_orig = nn.Parameter(torch.from_numpy(np.reshape(weight[start:stop],weight_shapes[2])).float())
    
    
    start = start+total
    total = np.prod(weight_shapes[3])
    stop = stop+total
    net.rnn.h2h.bias = nn.Parameter(torch.from_numpy(np.reshape(weight[start:stop],weight_shapes[3])).float())
    
    start = start+total
    total = np.prod(weight_shapes[4])
    stop = stop+total
    net.fc.weight = nn.Parameter(torch.from_numpy(np.reshape(weight[start:stop],weight_shapes[4])).float())
    
    start = start+total
    total = np.prod(weight_shapes[5])
    stop = stop+total
    net.fc.bias = nn.Parameter(torch.from_numpy(np.reshape(weight[start:stop],weight_shapes[5])).float())
    
    
    return net, all_weights
    


def train_multitask2(tasks,steps,mask,lr,randomize_task_order):
    #total possible tasks is 14 with this function
    """function to train the model on multiple tasks.
   
    Args:
        net: a pytorch nn.Module module
        dataset: a dataset object that when called produce a (input, target output) pair
   
    Returns:
        net: network object after training
    """
    if torch.cuda.is_available(): 
        dev = "cuda:0" 
        print(dev)
    else: 
        dev = "cpu" 
        print(dev)
    device = torch.device(dev) 
   

    

    # set tasks
    dt = 100
    #tasks = ["SineWavePred-v0","GoNogo-v0","PerceptualDecisionMaking-v0"]
    
    #tasks = ["GoNogo-v0","PerceptualDecisionMaking-v0"]
    num_tasks = len(tasks)
    kwargs = {'dt': dt}
    seq_len = 100

    # Make supervised datasets
    i1 = np.zeros([num_tasks])
    o1 = np.zeros([num_tasks])
    dataset1 = {}
    for task in range(num_tasks):
      
        dataset1[task] = ngym.Dataset(tasks[task], env_kwargs=kwargs, batch_size=16,
                       seq_len=seq_len)
       
                   #get input and output sizes for different tasks
        env = dataset1[task].env
        i1[task] = env.observation_space.shape[0]
        try:
            o1[task] = env.action_space.n
        except:
            o1[task] = env.action_space.shape[0]


    input_size = int(np.sum(i1))
    output_size = int(np.sum(o1))
    
    
    hidden_size = mask.shape[0]
   
   
    net = RNNNet(input_size=input_size, hidden_size=hidden_size,
             output_size=output_size, dt=dt)
             
    net.to(device)
   
        #apply pruning mask
    mask = torch.from_numpy(mask).to(device)
    apply = prune.custom_from_mask(net.rnn.h2h, name = "weight", mask = mask)
   
    # Use Adam optimizer
    optimizer = optim.Adam(net.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    running_loss = 0
    running_acc = 0
    start_time = time.time()
    # Loop over training batches
    #print('Training network...')
    num_param = 0
    num_param += net.rnn.input2h.weight.cpu().detach().numpy().size
    num_param += net.rnn.input2h.bias.cpu().detach().numpy().size
    num_param += net.rnn.h2h.weight.cpu().detach().numpy().size
    num_param += net.rnn.h2h.bias.cpu().detach().numpy().size
    num_param += net.fc.weight.cpu().detach().numpy().size
    num_param += net.fc.bias.cpu().detach().numpy().size

   
    loss_trajectory = np.zeros([steps])
    weight_trajectory = np.zeros((steps+1,num_param))
    
   
    for i in range(steps):
        # Generate input and target(labels) for all tasks, then concatenate and convert to pytorch tensor
        weight_trajectory[i,:] = weight_shape_to_flat(net)
        
        inputs1 = {}
        labels1 = {}
        for task in range(num_tasks):
            data = dataset1[task]
            inputs1[task], labels1[task] = data()
            
       
       
       
       
       

        #keep track of number of output neurons while stacking and change output labels so they are unique
        ####need to change labels so that they correspond to proper neuron output
        num_out_cumsum = 0
        for task in range(num_tasks):
       
            if task != 0:
       
       
                num_out_cumsum = num_out_cumsum + o1[task-1]

       
                idd = labels1[task] > 0
           
           
               
                labels1[task][idd] = labels1[task][idd]+num_out_cumsum
               
        rand_list = np.random.permutation(num_tasks)
        #now stack them
                #Here is where you could change the order of the tasks
        #Currently random ordering
       
        if randomize_task_order == 1:
            for task in range(num_tasks):

                if task == 0:
                    labels = labels1[rand_list[task]]
                else:
                    labels = np.concatenate((labels, labels1[rand_list[task]]), axis=0)


        else:
            for task in range(num_tasks):

                if task == 0:
                    labels = labels1[task]
                else:
                    labels = np.concatenate((labels, labels1[task]), axis=0)


       
        labels = torch.from_numpy(labels.flatten()).type(torch.long)
        #plt.hist(labels)
        #plt.show()
   
        ###Need to concatenate inputs along sequence length so that the tasks are given to the network sequentially
        #make same size on axis 3
        before = 0
        after = int(np.asarray(input_size)-i1[0])

        inputs1[0] = np.pad(inputs1[0],((0,0),(0,0),(before,after)))

       
        for task in range(num_tasks):
           
            if task != 0:
                before += int(i1[task-1])
                after = int(after-i1[task])
               
                inputs1[task] = np.pad(inputs1[task],((0,0),(0,0),(before,after)))

               

        #Here is where you could change the order of the tasks
        #Currently random ordering
       
        if randomize_task_order == 1:
           

            for task in range(num_tasks):
                if task == 0:
                    inputs = inputs1[rand_list[task]] #rand_list[task]
                else:
                    inputs = np.concatenate((inputs,inputs1[rand_list[task]]), axis=0)

        else:
            for task in range(num_tasks):
                if task == 0:
                    inputs = inputs1[task] #rand_list[task]
                else:
                    inputs = np.concatenate((inputs,inputs1[task]), axis=0)

           
        inputs = torch.from_numpy(inputs).type(torch.float).to(device)
       
       

       

       
        # boiler plate pytorch training:
        optimizer.zero_grad()   # zero the gradient buffers
        output, _ = net(inputs)
        # Reshape to (SeqLen x Batch, OutputSize)
        output = output.view(-1, output_size)
       
        #print("output shape: " + str(output.shape))
        #print("labels shape: " + str(labels.shape))
       
        
        loss = criterion(output, labels)
        
        loss.backward()
        optimizer.step()    # Does the update

        loss_trajectory[i] = loss.item()
       
        # Compute the running loss every 100 steps
        running_loss += loss.item()
        if i % 100 == 99:
            running_loss /= 100
            print('Step {}, Loss {:0.4f}, Time {:0.1f}s'.format(
                i+1, running_loss, time.time() - start_time))
            running_loss = 0
    return net, loss_trajectory, weight_trajectory



def get_loss(net,tasks):
    
    criterion = nn.CrossEntropyLoss()
    
    randomize_task_order = 1
    dt = 100
    num_tasks = len(tasks)
    kwargs = {'dt': dt}
    seq_len = 100

    # Make supervised datasets
    i1 = np.zeros([num_tasks])
    o1 = np.zeros([num_tasks])
    dataset1 = {}
    for task in range(num_tasks):
        dataset1[task] = ngym.Dataset(tasks[task], env_kwargs=kwargs, batch_size=16,
                       seq_len=seq_len)
       
                   #get input and output sizes for different tasks
        env = dataset1[task].env
        i1[task] = env.observation_space.shape[0]
        o1[task] = env.action_space.n


    input_size = int(np.sum(i1))
    output_size = int(np.sum(o1))
    
    inputs1 = {}
    labels1 = {}
    for task in range(num_tasks):
        data = dataset1[task]
        inputs1[task], labels1[task] = data()







    #keep track of number of output neurons while stacking and change output labels so they are unique
    ####need to change labels so that they correspond to proper neuron output
    num_out_cumsum = 0
    for task in range(num_tasks):

        if task != 0:


            num_out_cumsum = num_out_cumsum + o1[task-1]


            idd = labels1[task] > 0



            labels1[task][idd] = labels1[task][idd]+num_out_cumsum

    rand_list = np.random.permutation(num_tasks)
    #now stack them
            #Here is where you could change the order of the tasks
    #Currently random ordering

    if randomize_task_order == 1:
        for task in range(num_tasks):

            if task == 0:
                labels = labels1[rand_list[task]]
            else:
                labels = np.concatenate((labels, labels1[rand_list[task]]), axis=0)


    else:
        for task in range(num_tasks):

            if task == 0:
                labels = labels1[task]
            else:
                labels = np.concatenate((labels, labels1[task]), axis=0)



    labels = torch.from_numpy(labels.flatten()).type(torch.long)
    #plt.hist(labels)
    #plt.show()

    ###Need to concatenate inputs along sequence length so that the tasks are given to the network sequentially
    #make same size on axis 3
    before = 0
    after = int(np.asarray(input_size)-i1[0])

    inputs1[0] = np.pad(inputs1[0],((0,0),(0,0),(before,after)))


    for task in range(num_tasks):

        if task != 0:
            before += int(i1[task-1])
            after = int(after-i1[task])

            inputs1[task] = np.pad(inputs1[task],((0,0),(0,0),(before,after)))



    #Here is where you could change the order of the tasks
    #Currently random ordering

    if randomize_task_order == 1:


        for task in range(num_tasks):
            if task == 0:
                inputs = inputs1[rand_list[task]] #rand_list[task]
            else:
                inputs = np.concatenate((inputs,inputs1[rand_list[task]]), axis=0)

    else:
        for task in range(num_tasks):
            if task == 0:
                inputs = inputs1[task] #rand_list[task]
            else:
                inputs = np.concatenate((inputs,inputs1[task]), axis=0)


    inputs = torch.from_numpy(inputs).type(torch.float)

    #print("input_shape: ",inputs.shape)
    output, _ = net(inputs)
    # Reshape to (SeqLen x Batch, OutputSize)
    output = output.view(-1, output_size)

    #print("output shape: " + str(output.shape))
    #print("labels shape: " + str(labels.shape))


    loss = criterion(output, labels)
    
    return loss



from sklearn.decomposition import PCA
import time

def get_loss_landscape(tasks,nbins,ndim,mask,get_specialization):
    
    iterations = 10
    steps = 1000
    #mask = np.ones((100,100))
    randomize_task_order = 1
    lr = 0.01
    #hidden_size = 100
    
    #sample_weights
    for iterr in range(iterations):
        print("iterations: ", iterr)
        net, loss_traj, weight_traj = train_multitask2(tasks,steps,mask,lr,randomize_task_order)

        if iterr == 0:
            total_weight_traj = weight_traj
        else:
            total_weight_traj = np.append(total_weight_traj, weight_traj, axis=0)
        
    
    
    #lower dimensionality, get PCs
    pca = PCA(n_components=ndim)
    pca.fit(total_weight_traj)
    pca_result = pca.transform(total_weight_traj)
   

    #Create a grid in the 2D PCA space
    #x_vals = np.linspace(min(pca_result[:, 0])-5, max(pca_result[:, 0])+5, nbins)
    #y_vals = np.linspace(min(pca_result[:, 1])-5, max(pca_result[:, 1])+5, nbins)
    #grid_points_pca = np.array([[x, y] for x in x_vals for y in y_vals])
    

    N = pca_result.shape[1]
    
    # Create an array of linearly spaced values for each dimension
    ranges = [np.linspace(min(pca_result[:, i])-5, max(pca_result[:, i])+5, nbins) for i in range(N)]
    
    # Create a mesh grid for N dimensions
    mesh = np.meshgrid(*ranges)
    
    # Reshape the mesh grid to create a list of grid points
    grid_points_pca = np.vstack(map(np.ravel, mesh)).T
    


    # Project points back to original space
    grid_points_original = pca.inverse_transform(grid_points_pca)

    explained_variance = pca.explained_variance_ratio_
    print(explained_variance)
    
    i = 0
    loss = np.zeros(nbins**ndim)
    all_weights = []
    specialization_Q = np.zeros(nbins**ndim)
    
    for weights in grid_points_original:
        
        if i % 10 == 0:
            print("i: ",i)
            
        net, WW = weight_flat_to_shape(net,weights)
        
        all_weights.append(WW)
        
        loss[i] = get_loss(net,tasks)
        
        
        if get_specialization == True:
            start_time = time.time()
            Acc, diff = specialization_quotient(net,tasks)
            
            #plt.plot(Acc)
            #plt.show()
            
            specialization_Q[i] = np.mean(diff)
            end_time = time.time()-start_time
            print("time: ", end_time)
        else:
            specialization_Q = []
        i += 1
            
        
    return grid_points_pca, loss, all_weights, explained_variance, specialization_Q, total_weight_traj



    




from sklearn.decomposition import PCA
import time
#import scipy.io as sio

def get_weight_trajectories(tasks,mask, save_name):
    
    iterations = 2000
    steps = 100
    #mask = np.ones((100,100))
    randomize_task_order = 1
    lr = 0.001
    #hidden_size = 100
    
    #sample_weights
    for iterr in range(iterations):
        print("iterations: ", iterr)
        start_time = time.time()
        net, loss_traj, weight_traj = train_multitask2(tasks,steps,mask,lr,randomize_task_order)

        if iterr == 0:
            total_weight_traj = weight_traj
        else:
            total_weight_traj = np.append(total_weight_traj, weight_traj, axis=0)
        
        if (iterr+1)%100 == 0:
            np.savez_compressed(save_name,total_weight_traj = total_weight_traj)
        
        print("time: ",time.time() - start_time)
        #mdic = {"total_weight_traj": total_weight_traj}

        #sio.savemat("total_weight_traj.mat", mdic)

    return total_weight_traj




from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
from scipy.stats import pearsonr
from scipy.optimize import linear_sum_assignment

def pca_validate(data,ndim,max_corr):
    
    num_steps = 100
    
    num_sample = data.shape[0]
    print("num_sample: ",num_sample)
    num_test = int(np.round(num_sample/2))
    
    test = data[0:num_test,:]
    train = data[num_test:num_test+num_test,:]

    # Scale data using RobustScaler
    robust_scaler = RobustScaler()
    X_scaled = robust_scaler.fit_transform(test)
    pca_test = PCA(n_components=ndim)
    pca_test.fit(X_scaled)
    test_components = pca_test.components_
    
    
    #change size of train and then correlate with test
    start = 0
    stop = num_steps
    all_corr = np.zeros((int(num_test/num_steps),ndim))
    for i in range(int(num_test/num_steps)):
        #print("start ",start,"stop ",stop)
        print("iteration: ",i)
        train1 = train[0:stop,:]
        # Scale data using RobustScaler
        robust_scaler = RobustScaler()
        X_scaled = robust_scaler.fit_transform(train1)
        pca_train = PCA(n_components=ndim)
        pca_train.fit(X_scaled)
        train_components = pca_train.components_
        
        
        if (i+1)%100 == 0:
            plt.plot(np.absolute(all_corr))
            plt.xlabel("number of iterations")
            plt.ylabel("similarity to other sample (N=500)")
            plt.title("So far...")
            plt.show()
            
        if max_corr == True:
            it = np.zeros((ndim,ndim))
            for j in range(ndim):
                for jj in range(ndim):
                    tt = np.corrcoef(train_components[j],test_components[jj])
                    it[j,jj] = tt[0,1]
            cost_matrix = 1 - np.absolute(it)
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            new_mat = 1 - cost_matrix[:,col_ind]
          
            #plt.imshow(new_mat)
            #plt.colorbar()
            #plt.show()

         
            
            all_corr[i,:] = np.diag(new_mat)
            
        else:
            for j in range(ndim):
                #print(train_components[j].shape)
                it = np.corrcoef(train_components[j],test_components[j])

                all_corr[i,j] = it[0,1]

        
        #print(it)
        stop = stop+num_steps
        
        
    return all_corr
    





