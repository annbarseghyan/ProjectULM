import networks
import loss_fns
import torch
from torch.autograd import Variable
import torch.optim as optim
import empirical_sim
import prior
import torch.nn as nn
import simple_noise
import generative_model
import time
import sys
import numpy as np
# Parameters
batch_size = 256//2#256*10
eval_batch_size = 256
iters_per_eval = 3#10#*4*4
stepsize = 1E-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_2D = True

if _2D:
    print("Using 2D loss.")

kernel_sigmas = [64.0, 320.0, 640.0, 1920.0]

#
# use_cuda = False
# warmstart = True

# Construct a generative model
# N changed from 5 to 10
p = prior.UniformCardinalityPrior([0,0], [6400,6400], 200.0, 1000.0, 30)
sim = empirical_sim.EmpiricalSim(64,6400,empirical_sim.load_AS())
# noise reduced from 100 to 50
noise = simple_noise.EMCCD(0)
gen_model = generative_model.GenerativeModel(p,sim,noise)

#Note that queues do not work on OS X.
# if sys.platform != 'darwin':
#     m = generative_model.MultiprocessGenerativeModel(gen_model,4,batch_size)
# else:

m = gen_model

# Construct the network
# Warmstart?
# if warmstart:
# else:
#     net = nets.DeepLoco() if not _2D else nets.DeepLoco(min_coords = [0,0], max_coords = [6400,6400])

net = networks.DeepLoco(min_coords = [0,0], max_coords = [6400,6400])
net = net.to(device)
net.load_state_dict(torch.load("adapted_data_simulation.pth"))

# if use_cuda:
#     net = net.cuda()

theta_mul = torch.Tensor([1.0,1.0]).to(device)

# if use_cuda:
#     theta_mul = theta_mul.cuda()

# Takes a CPU batch and converts to CUDA variables
# Also zero/ones the simulated weights
def to_device(d):
    return torch.tensor(d, device=device)

def to_variable(theta, weights, images):
    return to_device(theta), to_device(weights).sign_(),to_device(images)

# Loss function
def loss_fn(o_theta, o_w, theta, weights):
    if _2D:
        theta = theta[:,:,:2]
    return loss_fns.multiscale_l1_laplacian_loss(o_theta*theta_mul, o_w,
                                                 theta*theta_mul, weights,
                                                 kernel_sigmas).mean()

# Generate an evaluation batch
(e_theta, e_weights, e_images) = m.sample_eval_batch(eval_batch_size,113)
(e_theta, e_weights, e_images) = to_variable(e_theta, e_weights, e_images)
np.save("deeploco_simulated_images.npy", e_images.cpu().numpy())

# lr_schedule = [(stepsize, 300), (stepsize/2, 200), (stepsize/4, 100) ,(stepsize/8, 100), (stepsize/16, 100),(stepsize/32, 100),(stepsize/64, 100)]
lr_schedule = [(stepsize, 500), (stepsize/2, 300), (stepsize/4, 200) ,(stepsize/8, 200), (stepsize/16, 100),(stepsize/32, 100),(stepsize/64, 100),(stepsize/128, 100)]


for stepsize, iters in lr_schedule:
    # Constuct the optimizer
    print("stepsize = ", stepsize)
    optimizer = optim.Adam(net.parameters(),lr=stepsize)
    for i in range(iters):
        iter_start_time = time.time()
        print("iter",i)
        # Compute eval
        (o_theta_e, o_w_e) = net(e_images.to(torch.float32))
        e_loss = loss_fn(o_theta_e, o_w_e, e_theta,e_weights)
        print("\teval", e_loss.item())

        s_time = time.time()
        for batch_idx in range(iters_per_eval):
            print(".")
            (theta, weights, images) = m.sample(batch_size)
            optimizer.zero_grad()
            theta, weights, images = to_variable(theta, weights, images)
            (o_theta, o_w) = net(images.to(torch.float32))
            train_loss = loss_fn(o_theta,o_w, theta, weights)
            train_loss.backward()
            optimizer.step()
        # torch.save(net, "net_AS")
        torch.save(net.state_dict(), "adapted_data_simulationV1.pth")
        # if use_cuda:
        #     net.cuda()
        print("A:", time.time()-iter_start_time)