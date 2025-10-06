import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
from collections import deque
import random
import matplotlib.pyplot as plt

# HYPER PARAMETERS
BATCH_SIZE = 100
MEMORY_SIZE_TOTAL = 1000000
MEMORY_SIZE_MIN = 100  # MIN MEMORY SIZE BEFORE OPTIMIZING (LEARNING)
DISCOUNT_FACTOR = 0.99  # Gamma
TAU = 0.005
LR = 0.001
WARMUP_STEPS = 1000
EXPLORE_NOISE = 0.1
POLICY_NOISE = 0.2
POLICY_FREQ = 2
NOISE_CLIP = 0.5


IS_HARDCORE = True
TOTAL_STEPS = 1000000
TEST_EPISODES = 1000


# SET TRAIN BOOL, SET VERSION NUMBER TO SAVE (MODEL VERSION)
TRAIN_BOOL = False
VERSION_CONTROL = "DOWN"  # Does not matter if TRAIN_BOOL is set to False (New version is only created when training)

# SET TEST BOOL, SET VERSION NUMBER TO LOAD IN TEST (MODEL)
TEST_BOOL = True
MODEL_VERSION_LOAD = "HARDCORE"  # Does not matter if TEST_BOOL is set to False (Loading a model is to test it)

# DEVICE ---> "cuda" if TRAIN_BOOL = True
#             "cpu" if TEST_BOOL = True
# TODO Currently run gpu on training -> fix it
DEVICE = torch.device("cpu")


# Actor
class Actor(nn.Module):
    def __init__(self, state_size, action_size, max_action):
        super(Actor, self).__init__()
        # Actor Architecture
        self.inp = nn.Linear(state_size, 300)
        self.fc1 = nn.Linear(300, 400)
        self.out = nn.Linear(400, action_size)

        self.max_action = max_action

    def forward(self, state):
        x = F.relu(self.inp(state))
        x = F.relu(self.fc1(x))
        x = self.max_action * F.tanh(
            self.out(x)
        )  # Multiplying with action is only needed, if the action space isnt within -1 to 1 (tanh)
        return x


# Critics
class Critics(nn.Module):
    def __init__(self, state_size, action_size):
        super(Critics, self).__init__()

        # Q1 Architecture
        self.inp = nn.Linear(state_size + action_size, 300)
        self.fc1 = nn.Linear(300, 400)
        self.out = nn.Linear(400, 1)

        # Q2 Architecture
        self.inp_2 = nn.Linear(state_size + action_size, 300)
        self.fc1_2 = nn.Linear(300, 400)
        self.out_2 = nn.Linear(400, 1)

    def forward(self, state, action):
        # Concat state + action
        sa = torch.cat([state, action], 1)

        # Q1 Forward
        q1 = F.relu(self.inp(sa))
        q1 = F.relu(self.fc1(q1))
        q1 = self.out(q1)

        # Q2 Forward
        q2 = F.relu(self.inp_2(sa))
        q2 = F.relu(self.fc1_2(q2))
        q2 = self.out_2(q2)

        return q1, q2

    def Q1(self, state, action):
        # Concat state + action
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.inp(sa))
        q1 = F.relu(self.fc1(q1))
        q1 = self.out(q1)
        return q1


# Define Replay Memory
class ReplayMemory:
    def __init__(self):
        self.memory = deque(maxlen=MEMORY_SIZE_TOTAL)

    def append(self, transition):
        self.memory.append(transition)

    def sample(self):
        batch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(states).float().to(DEVICE)
        actions = torch.tensor(actions).float().to(DEVICE)
        rewards = torch.tensor(rewards).float().to(DEVICE)
        next_states = torch.tensor(next_states).float().to(DEVICE)
        dones = torch.tensor(dones).float().to(DEVICE)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)


class TD3:
    def __init__(self):
        # Setup parameters
        self.env = gym.make("BipedalWalker-v3", hardcore=IS_HARDCORE)
        self.min_action = self.env.action_space.low
        self.max_action = self.env.action_space.high
        self.max_action_t = torch.tensor(self.env.action_space.high).float().to(DEVICE)
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.shape[0]
        self.warmup_steps = WARMUP_STEPS

        # ReplayMemory
        self.memory = ReplayMemory()

        # Actor Setup
        self.actor = Actor(self.state_size, self.action_size, self.max_action_t).to(
            DEVICE
        )
        # Actor Target Setup
        self.actor_target = Actor(
            self.state_size, self.action_size, self.max_action_t
        ).to(DEVICE)
        # Set actor optimizer
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=LR)

        # Critics Setup
        self.critics = Critics(self.state_size, self.action_size).to(
            DEVICE
        )  # only needs state + action
        # Critics Target Setup
        self.critics_target = Critics(self.state_size, self.action_size).to(DEVICE)
        # Sets same weights from critics to critics target
        self.critics_target.load_state_dict(self.critics.state_dict())
        # Set critic optimizer
        self.critics_optimizer = torch.optim.Adam(self.critics.parameters(), lr=LR)

    def select_action(self, state, noise=0.1):
        # Get best action based on current policy
        state = torch.tensor(state.reshape(1, -1)).float().to(DEVICE)
        action = self.actor(state).cpu().data.numpy().flatten()

        return action

    def train(self):
        seed_array = [
            157,
            172,
            187,
            230,
            250,
            265,
            323,
            382,
            454,
            479,
            643,
            675,
            689,
            702,
            1130,
            1137,
            1183,
            1054,
            1254,
            1286,
            1306,
        ]
        seed_counter = 0
        reward_history = []
        episodes_done = 0
        state = self.env.reset()
        done = False
        episode_reward = 0
        episode_steps = 0

        for steps_done in range(TOTAL_STEPS):

            # Select action
            if self.warmup_steps > steps_done:
                action = self.env.action_space.sample()

            else:
                action = (
                    self.select_action(np.array(state))
                    + np.random.normal(0, EXPLORE_NOISE, size=self.action_size)
                ).clip(-self.max_action, self.max_action)

            # Execute actions
            next_state, reward, done, info = self.env.step(action)

            if reward <= -100:
                reward = -5

            # Append transition to memory
            self.memory.append((state, action, reward, next_state, done))

            # Update state, Update reward, Update episode steps
            state = next_state
            episode_reward += reward
            episode_steps += 1

            # Optimize if replay memory is large enough
            if len(self.memory) > MEMORY_SIZE_MIN:
                self.optimize(episode_steps)

            if done:
                episodes_done += 1

                reward_history.append(episode_reward)

                print(
                    f"Episode: {episodes_done} --- Reward: {episode_reward} --- Current Seed: NOTHING --- ES/TS: {episode_steps} / {steps_done}"
                )

                seed_counter += 1
                if seed_counter == len(seed_array):
                    seed_counter = 0

                state = self.env.reset(seed=seed_array[seed_counter])

                done = False
                episode_reward = 0
                episode_steps = 0

        self.env.close()

        # Save Actor Network
        torch.save(
            self.actor.state_dict(),
            "TD3/loadModels/bipedalTD3{}.pth".format(VERSION_CONTROL),
        )

        # Create Graph
        plt.figure(1)

        # Plot the Reward History
        plt.plot(reward_history)
        plt.title("Reward History")
        plt.xlabel("Episodes")
        plt.ylabel("Reward")

        # Save Graph
        plt.savefig("TD3/graphs/bipedalTD3{}.png".format(VERSION_CONTROL))

    def optimize(self, current_iteration):

        state, action, reward, next_state, done = self.memory.sample()

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (torch.randn_like(action) * POLICY_NOISE).clamp(
                -NOISE_CLIP, NOISE_CLIP
            )

            next_action = (self.actor_target(next_state) + noise).clamp(
                -self.max_action_t, self.max_action_t
            )

            # Compute the target Q value
            target_Q1, target_Q2 = self.critics_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)

            done = done.unsqueeze(1)
            reward = reward.unsqueeze(1)

            target_Q = reward + (1 - done) * DISCOUNT_FACTOR * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critics(state, action)

        # Compute critic loss
        critics_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
            current_Q2, target_Q
        )

        # Optimize the critic
        self.critics_optimizer.zero_grad()
        critics_loss.backward()
        self.critics_optimizer.step()

        if current_iteration % POLICY_FREQ == 0:
            # Compute actor loss
            actor_loss = -self.critics.Q1(state, self.actor(state)).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(
                self.critics.parameters(), self.critics_target.parameters()
            ):
                target_param.data.copy_(
                    TAU * param.data + (1 - TAU) * target_param.data
                )
            for param, target_param in zip(
                self.actor.parameters(), self.actor_target.parameters()
            ):
                target_param.data.copy_(
                    TAU * param.data + (1 - TAU) * target_param.data
                )

    def test(self, episodes):
        self.env = gym.make(
            "BipedalWalker-v3",
            hardcore=IS_HARDCORE,
            render_mode="human" if TEST_BOOL else None,
        )

        seed_array = [
            184,
            311,
            356,
            483,
            507,
            612,
            638,
            651,
            665,
            669,
            680,
            688,
            706,
            1052,
            1301,
            1309,
            1314,
            1206,
        ]

        if TEST_BOOL:
            self.actor = Actor(self.state_size, self.action_size, self.max_action_t)
            self.actor.load_state_dict(
                torch.load("TD3/loadModels/bipedalTD3{}.pth".format(MODEL_VERSION_LOAD))
            )
            self.actor.eval()
        else:
            avg_reward = 0

        for i in range(episodes):
            state = self.env.reset()
            done = False
            episode_reward = 0

            while not done:
                action = self.select_action(np.array(state))
                state, reward, done, info = self.env.step(action)
                episode_reward += reward

            if TEST_BOOL:
                print(
                    f"Episode {i+1} Reward: {episode_reward}, Seednumber: {seed_array[i]}"
                )
            else:
                avg_reward += episode_reward

        if TEST_BOOL:
            self.env.close()
        else:
            avg_reward /= episodes
            return avg_reward


# CUDA CHECKS
print(torch.cuda.is_available())
print(torch.cuda.current_device())
current_device = torch.cuda.current_device()
print(torch.cuda.get_device_name(current_device))

agent = TD3()
if TRAIN_BOOL:
    agent.train()
if TEST_BOOL:
    agent.test(TEST_EPISODES)
