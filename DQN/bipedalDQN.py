import gym
import numpy as np
import matplotlib.pyplot as plt 
from collections import deque
import random
import torch
from torch import nn
import torch.nn.functional as F

#HYPER PARAMETERS
BATCH_SIZE = 100
MEMORY_SIZE_TOTAL = 5000
MEMORY_SIZE_MIN = 100
DISCOUNT_FACTOR = 0.99 #Gamma
EPSILON_START = 1
EPSILON_DECAY = 0.0011
EPSILON_END = 0.05
OPTIMIZE_RATE = 4 #How often we optimize the model (unit: steps)
TARGET_SYNC_RATE = 1 #How often we update the target network (unit: episodes)
LR = 0.001
IS_HARDCORE = False
EPISODES = 1000
TEST_EPISODES = 10

#SET TRAIN BOOL, SET VERSION NUMBER TO SAVE (MODEL VERSION), SET TEST BOOL, SET VERSION NUMBER TO LOAD IN TEST (MODEL) 
TRAIN_BOOL = False
VERSION_CONTROL = '03'  #Does not matter if TRAIN_BOOL is set to False (New version is only created when training)

TEST_BOOL = True
MODEL_VERSION_LOAD = '03' #Does not matter if TEST_BOOL is set to False (Loading a model is to test it)

# Define Model
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.inp = nn.Linear(state_size, 64)
        self.fc1 = nn.Linear(64, 32)
        self.out = nn.Linear(32, action_size)
        
    def forward(self, x):
        x = F.relu(self.inp(x))
        x = F.relu(self.fc1(x))
        x = F.tanh(self.out(x))
        return x
        
        
# Define Replay Memory
class ReplayMemory:
    def __init__(self):
        self.memory = deque(maxlen=MEMORY_SIZE_TOTAL)
        
    def append(self, transition):
        self.memory.append(transition)
        
    def sample(self):
        return random.sample(self.memory, BATCH_SIZE)
    
    def __len__(self):
        return len(self.memory)
    
    
# Define Agent
class BipedalDQN: 
   
    optimizer = None
    
    def train (self, episodes, render):
        
        # Grab the environment + state and action size
        env = gym.make('BipedalWalker-v3', hardcore=IS_HARDCORE, render_mode='human' if render else None)
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.shape[0]
        
        # Initialize Epsilon + Memory
        self.epsilon = EPSILON_START
        self.memory = ReplayMemory()
        
        # Create policy and target networks
        self.policy_net = DQN(state_size, action_size)
        self.target_net = DQN(state_size, action_size) 
        
        # Copy the weights from the policy network to the target network
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Initialize the optimizer
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=LR)
        
        # Allocate space for list of rewards
        reward_history = []
        
        # List to track epsilon decay
        epsilon_history = []
        
        # Steps variables to track sync, episode steps and total steps
        self.episodes_since_last_sync = 0
        self.steps_since_last_optimize = 0
        self.episode_steps = 0
        self.total_steps = 0
        
        
        for i in range(episodes):
            state = env.reset()
            done = False
            #truncated = False
            episode_reward = 0
            
            while(not done):
                
                # Choose random action
                if random.random() < self.epsilon:
                    action = env.action_space.sample()
                
                
                # Choose best action
                else:
                    with torch.no_grad():
                        action = self.policy_net(torch.tensor(state)).float()

                # Execute action
                
                next_state, reward, done, info = env.step(action)
                
                self.episode_steps += 1
                self.steps_since_last_optimize += 1
                episode_reward += reward
                
                # Try to remove local minima
                if (episode_reward < -20) & (self.episode_steps == 1200):
                    reward += -100
                    episode_reward += -100
                    done = True

                # Append to memory
                action = torch.tensor(action).long()
                self.memory.append((state, action, reward, next_state, done))
                
                # Update state
                state = next_state
                
                
                if (len(self.memory) > MEMORY_SIZE_MIN) & (self.steps_since_last_optimize >= OPTIMIZE_RATE):
                    # Sample a mini-batch from the replay memory and optimize
                    mini_batch = self.memory.sample()
                    self.optimize(mini_batch, self.policy_net, self.target_net)
                    self.steps_since_last_optimize = 0
                
        
            reward_history.append(episode_reward)
            self.total_steps += self.episode_steps
            print (f"Episode: {i+1} / {episodes} --- Reward: {episode_reward} --- ES/TS: {self.episode_steps} / {self.total_steps} --- Epsilon: {self.epsilon}")
            self.episode_steps = 0
            
            self.episodes_since_last_sync += 1
            if self.episodes_since_last_sync % TARGET_SYNC_RATE == 0:
                # Update target network
                self.target_net.load_state_dict(self.policy_net.state_dict())
                self.target_net.eval()
            
            # POSSIBLE ---> if episode_reward > self.high_score:
            # Decay epsilon
            self.epsilon = max(EPSILON_END, self.epsilon - EPSILON_DECAY)
            epsilon_history.append(self.epsilon)
                
                
        env.close()
        
        torch.save(self.policy_net.state_dict(), 'DQN/loadModels/bipedalDQN{}.pth'.format(VERSION_CONTROL))
        
        # Create Graph 
        plt.figure(1)
        
        # Plot the reward history
        plt.subplot(211)
        plt.plot(reward_history)
        plt.title('Reward History')
        
        # Plot the epsilon history
        plt.subplot(212)
        plt.plot(epsilon_history)
        plt.title('Epsilon History')
        
        plt.savefig('DQN/graphs/bipedalDQN{}.png'.format(VERSION_CONTROL))
        
                        
    def optimize(self, mini_batch, policy_net, target_net):
        current_q_list = []
        target_q_list = []
        
        
        for state, action, reward, next_state, done in mini_batch:
            
            # if done ? target = reward : target = reward + gamma * max_a Q(s', a)
            with torch.no_grad():
                target_q = reward + DISCOUNT_FACTOR * target_net(torch.tensor(next_state)).float() * (1-done)
            
            current_q = policy_net(torch.tensor(state))
            current_q_list.append(current_q)
            
            target_q_list.append(target_q)  
                
        
        # Compute loss for all transitions in batch
        loss = F.mse_loss(torch.stack(current_q_list), torch.stack(target_q_list))
        
        # Optimize the model     
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
     
    def test(self, episodes):
        
        env = gym.make('BipedalWalker-v3', hardcore=IS_HARDCORE, render_mode='human')
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.shape[0]
        
        self.policy_net = DQN(state_size, action_size)
        self.policy_net.load_state_dict(torch.load('DQN/loadModels/bipedalDQN{}.pth'.format(MODEL_VERSION_LOAD)))
        self.policy_net.eval()
        
        for i in range(episodes):
            state = env.reset()
            done = False
            #truncated = False
            episode_reward = 0
            
            while(not done):
                with torch.no_grad():
                    action = self.policy_net(torch.tensor(state)).float()
                    
                state, reward, done, info = env.step(action)
                episode_reward += reward
            
            print(f"Episode {i+1} Reward: {episode_reward}")    
            
        env.close()
          
          
agent = BipedalDQN()
if TRAIN_BOOL: agent.train(EPISODES, False)
if TEST_BOOL: agent.test(TEST_EPISODES)
        
        
                
                    
                
                
                
            
                
        
       
        
        
        
        
        
        
    
    
    