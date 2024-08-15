import numpy as np 
import torch
import torch.nn as nn
from collections import namedtuple, deque
import random
Transition = namedtuple('Transition',
                        ('state', 'action', 'done', 'next_state', 'reward')
                        )
#function to initialize the wts that are distributed uniformly acc to the variance of the wts being inversely prop. to the size of tensor
device = 'cuda' if torch.cuda.is_available() else 'cpu'
def fan_init_weights(tensor): 
    size = tensor.size(-1)
    variance = 1/np.sqrt(size)
    
    nn.init.uniform_(tensor, -variance, variance)
    
class Actor(nn.Module):
    def __init__(self, num_inputs, action_space):
        super(Actor, self).__init__()
        input_size = num_inputs
        output_size = action_space.shape[0]
        self.sequential = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, output_size),
            nn.Tanh()
        )
        
        #initialize the weights
        fan_init_weights(self.sequential[0].weight)
        fan_init_weights(self.sequential[0].bias)
        fan_init_weights(self.sequential[3].weight)
        fan_init_weights(self.sequential[3].bias)
        
    def forward(self, x):
        out = self.sequential(x)
        
        return out
    
    def add_noise(self, scalar = 0.1):
        for layer in [0, 3, 5]:
            self.sequential[layer].weight.data += torch.randn_like(self.sequential[layer].weight.data)*scalar
        
    
class Critic(nn.Module):
    def __init__(self, num_inputs, action_space):
        super(Critic, self).__init__()
        input_size = num_inputs
        
        output_size = action_space.shape[0]
        
        self.fc1 = nn.Linear(input_size, 128)
        self.norm1 = nn.LayerNorm(128)
        
        self.fc2 = nn.Linear(128+output_size, 128)
        self.norm2 = nn.LayerNorm(128)
        
        self.fc3 = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        #initialize the weights
        fan_init_weights(self.fc1.weight)
        fan_init_weights(self.fc1.bias)
        fan_init_weights(self.fc2.weight)
        fan_init_weights(self.fc2.bias)
        
    def forward(self, x, a):
        print("critic")
        out = self.fc1(x)
        out = self.norm1(out)        
        out = self.relu(out)
        print(out.size())
        out = torch.cat([out, a], 1) #to concatenate the two layers
        print(out.size)
        out = self.fc2(out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.tanh(out)
        
        return  out
        
class AdaptiveParamNoise(object):
    def __init__(self, initial_stddev = 0.1, desired_stddev = 0.2, adaptation_coeff = 1.01):
        self.initial_stddev = initial_stddev
        self.desired_stddev = desired_stddev
        self.adaptation_coeff = adaptation_coeff
        self.current_stddev = initial_stddev
        
    def adapt(self, distance):
        if distance > self.desired_stddev:
            self.current_stddev /= adaptation_coeff
        else:
            self.current_stddev *= adaptation_coeff
    
    def get_stats(self):
        stats = {
            'param_noise_stddev': self.current_stddev,
        }
        return stats

def distance(action_1, action_2):
    diff = action2 - action_1
    mean_diff = np.mean(np.sqr(diff), axis = 0 )
    distance = np.sqrt(np.mean(mean_diff))
    return distance
    
class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def noise(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt \
            + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)