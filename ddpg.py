import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam 
from model import (Actor, Critic, OrnsteinUhlenbeckActionNoise, ReplayMemory)
import torch.nn.functional as Loss


device = 'cuda' if torch.cuda.is_available() else 'cpu'
def softUpdate(network, target_network, tau):
    for target_param, param in zip(target_network.parameters(), network.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

class DDPG(object):
    def __init__(self, num_inputs, action_space, gamma, tau):
        self.gamma = gamma
        self.tau = tau
        self.action_space = action_space
        self.num_inputs = num_inputs
        print("hello")
        self.actor = Actor(self.num_inputs, self.action_space).to(device)
        self.actor_target = Actor(self.num_inputs, self.action_space).to(device)
        
        self.critic = Critic(self.num_inputs, self.action_space).to(device)
        self.critic_target = Critic(self.num_inputs, self.action_space).to(device)
        
        self.actor_optimizer = Adam(self.actor.parameters(), lr = 0.1)
        self.critic_optimizer = Adam(self.critic.parameters(), lr = 0.1)
        
        #hard update initially to ensure both the networks have the same parameters:
        for actor_target_param, actor_param in zip(self.actor_target.parameters(), self.actor.parameters()):
            actor_target_param.data.copy_(actor_param.data) #Copy the network params to target to ensure that both have the same params at starting
            
        for critic_target_param, critic_param in zip(self.critic_target.parameters(), self.critic.parameters()):
            critic_target_param.data.copy_(critic_param.data)
        
        #Have to work on adding the code to save the learned models.
        
        
    def find_action(self, state, ou_noise):
        input_state = state.to(device)
        self.actor.eval()
        action = self.actor(input_state)
        self.actor.train()
        action = action.data
        noised_action = action + torch.Tensor(ou_noise.noise()).to(device)
        noised_action = noised_action.clamp(self.action_space.low[0], self.action_space.high[0])
        
        return noised_action
    
    def update_params(self, batch):
        #get the batch values:
        state_batch = torch.cat(batch.state).to(device)
        action_batch = torch.cat(batch.action).to(device)
        done_batch = torch.cat(batch.done).to(device)
        reward_batch = torch.cat(batch.reward).to(device)
        next_state_batch = torch.cat(batch.next_state).to(device)

        next_action_batch = self.actor_target(state_batch)
        print("next_action_batch")
        
        next_value = self.critic_target(state_batch, next_action_batch.detach())
        print("next value - ", next_value)
        reward_batch = reward_batch.unsqueeze(1)
        done_batch = done_batch.unsqueeze(1)
        print("Done batch", done_batch.size())
        expected_value = reward_batch + (1 - done_batch)*self.gamma*next_value
        print("expected value", expected_value.size())
        #updating the critic network
        self.critic_optimizer.zero_grad() 
        state_action_batch = self.critic(state_batch, action_batch)
        print("state_action", state_action_batch.size())
        value_loss = Loss.mse_loss(state_action_batch, expected_value.detach())
        value_loss.backward()
        self.critic_optimizer.step()   

        #updating the policy network
        self.actor_optimizer.zero_grad() 
        policy_loss = -self.critic(state_batch, self.actor(state_batch))
        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.actor_optimizer.step()  
        
        #soft update of both the target networks
        softUpdate(self.critic_target, self.critic, self.tau)
        softUpdate(self.actor_target, self.actor, self.tau)
        return value_loss, policy_loss
        
    
