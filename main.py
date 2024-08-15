import torch
import gym
import gym_jsbsim
import numpy as np
from torch.utils.tensorboard import SummaryWriter


from ddpg import DDPG
from model import ReplayMemory, Transition, OrnsteinUhlenbeckActionNoise

device = 'cuda' if torch.cuda.is_available() else 'cpu'
env = gym.make('JSBSim-TurnHeadingControlTask-Cessna172P-Shaping.STANDARD-NoFG')

gamma = 0.99
tau = 0.01
hidden_size = tuple([64, 64])
memory_size = int(1e5)
time_steps = int(1e6)
batch_size = 64
writer = SummaryWriter()

env.seed(0)
print(env.observation_space)
agent = DDPG(17, env.action_space, gamma, tau)

memory = ReplayMemory(memory_size)
nb_actions = env.action_space.shape[-1]
ou_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(nb_actions), sigma=float(0.2) * np.ones(nb_actions))

start_step = 0
time_step = start_step//10000 + 1
rewards, policy_loss, value_loss, mean_test_reward = [], [], [], []
epoch = 0
t = 0

while time_step <= time_steps:
    ou_noise.reset()
    value_losses = []
    policy_losses = []
    epoch_return = 0
    state = torch.Tensor([env.reset()]).to(device)
    while True:
        action = DDPG.find_action(agent, state=state, ou_noise=ou_noise)
        step_action = np.array(action)
        next_state, reward, done, info = env.step(step_action[0])
        time_step += 1
        epoch_return += reward
        
        mask = torch.Tensor([done]).to(device)
        reward = torch.Tensor([reward]).to(device)
        next_state = torch.Tensor([next_state]).to(device)
        
        memory.push(state, action, mask, next_state, reward)        
        state = next_state
        
        epoch_policy_loss, epoch_value_loss = 0, 0
        
        if len(memory) > batch_size:
            transitions = memory.sample(64)
            batch = Transition(*zip(*transitions))
            
            value_loss, policy_loss = DDPG.update_params(agent, batch)
            epoch_value_loss += value_loss
            epoch_policy_loss += policy_loss
        
        if done:
            break
    rewards.append(epoch_return)
    value_losses.append(epoch_value_loss)
    policy_losses.append(epoch_policy_loss)
    writer.add_scalar('epoch/return', epoch_return, epoch)
    
    if time_step >= 10000*t:
        t += 1
        test_rewards = []
        mean_test_rewards = []
        for _ in range(10):  
            test_reward = 0
            state = torch.Tensor([env.reset()]).to(device)          
            while True:
                test_action = DDPG.find_action(agent, state, ou_noise)
                step_test_action = np.array(test_action)
                next_state, reward, done, info = env.step(step_test_action[0])
                test_reward += reward
                next_state = torch.Tensor([next_state]).to(device)
                state = next_state
                
                if done:
                    break
            test_rewards.append(test_reward)
            
        mean_test_rewards.append(np.mean(test_rewards))   
        writer.add_scalar('test/mean_test_return', mean_test_rewards[-1], epoch)
        
env.close()