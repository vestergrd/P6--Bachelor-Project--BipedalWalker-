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
CHALLENGE_AMOUNT = 5


IS_HARDCORE = True
TOTAL_STEPS = 1000000
TEST_EPISODES = 5


# SET TRAIN BOOL, SET VERSION NUMBER TO SAVE (MODEL VERSION)
TRAIN_BOOL = True
VERSION_CONTROL = "10-100"  # Does not matter if TRAIN_BOOL is set to False (New version is only created when training)

# SET TEST BOOL, SET VERSION NUMBER TO LOAD IN TEST (MODEL)
TEST_BOOL = False
MODEL_VERSION_LOAD = (
    "06"  # Does not matter if TEST_BOOL is set to False (Loading a model is to test it)
)

# DEVICE ---> "cuda" if TRAIN_BOOL = True
#             "cpu" if TEST_BOOL = True
# TODO Currently run gpu on training -> fix it
DEVICE = torch.device("cuda")


# Actor
class ChallengePredictActor(nn.Module):
    def __init__(self, lidar_data_size, challenges_amount):
        super(ChallengePredictActor, self).__init__()
        # Actor Architecture
        self.inp = nn.Linear(lidar_data_size, 300)
        self.fc1 = nn.Linear(300, 400)
        self.out = nn.Linear(400, challenges_amount)

    def forward(self, state):
        x = F.relu(self.inp(state))
        x = F.relu(self.fc1(x))
        x = F.tanh(self.out(x))
        return x


# Critics
class ChallengePredictCritics(nn.Module):
    def __init__(self, lidar_data_size, challenges_amount):
        super(ChallengePredictCritics, self).__init__()

        # Q1 Architecture
        self.inp = nn.Linear(lidar_data_size + challenges_amount, 300)
        self.fc1 = nn.Linear(300, 400)
        self.out = nn.Linear(400, 1)

        # Q2 Architecture
        self.inp_2 = nn.Linear(lidar_data_size + challenges_amount, 300)
        self.fc1_2 = nn.Linear(300, 400)
        self.out_2 = nn.Linear(400, 1)

    def forward(self, lidar_data, challenge_data):
        # Concat lidar_data + challenge_data
        sa = torch.cat([lidar_data, challenge_data], 1)

        # Q1 Forward
        q1 = F.relu(self.inp(sa))
        q1 = F.relu(self.fc1(q1))
        q1 = self.out(q1)

        # Q2 Forward
        q2 = F.relu(self.inp_2(sa))
        q2 = F.relu(self.fc1_2(q2))
        q2 = self.out_2(q2)

        return q1, q2

    def Q1(self, lidar_data, challenge_data):
        # Concat lidar_data + challenge_data
        sa = torch.cat([lidar_data, challenge_data], 1)

        q1 = F.relu(self.inp(sa))
        q1 = F.relu(self.fc1(q1))
        q1 = self.out(q1)
        return q1


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


# Define Replay Memory
class ReplayMemory:
    def __init__(self):
        self.memory = deque(maxlen=MEMORY_SIZE_TOTAL)

    def append(self, transition):
        self.memory.append(transition)

    def sample(self):
        batch = random.sample(self.memory, BATCH_SIZE)
        (
            states,
            actions,
            rewards,
            next_states,
            dones,
            pick_best_actor_data,
            lidar_data,
        ) = zip(*batch)

        states = torch.tensor(states).float().to(DEVICE)
        actions = torch.tensor(actions).float().to(DEVICE)
        rewards = torch.tensor(rewards).float().to(DEVICE)
        next_states = torch.tensor(next_states).float().to(DEVICE)
        dones = torch.tensor(dones).float().to(DEVICE)
        pick_best_actor_data = torch.tensor(pick_best_actor_data).float().to(DEVICE)
        lidar_data = torch.tensor(lidar_data).float().to(DEVICE)

        return (
            states,
            actions,
            rewards,
            next_states,
            dones,
            pick_best_actor_data,
            lidar_data,
        )

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
        self.lidar_data_size = 10
        self.challenge_amount = CHALLENGE_AMOUNT

        # ReplayMemory
        self.memory = ReplayMemory()
        ### ACTOR DECIDER
        # actor decider setup
        self.challenge_predict_actor = ChallengePredictActor(
            self.lidar_data_size, self.challenge_amount
        ).to(DEVICE)

        # Actor decider Target Setup
        self.challenge_predict_actor_target = ChallengePredictActor(
            self.lidar_data_size, self.challenge_amount
        ).to(DEVICE)

        # Optimizer for Actor_decider
        self.challenge_predict_actor_optimizer = torch.optim.Adam(
            self.challenge_predict_actor.parameters(), lr=LR
        )

        # actor_decider critics setup
        self.challenge_predict_critics = ChallengePredictCritics(
            self.lidar_data_size, self.challenge_amount
        ).to(DEVICE)

        # actor_decider Critics Target Setup
        self.challenge_predict_critics_target = ChallengePredictCritics(
            self.lidar_data_size, self.challenge_amount
        ).to(DEVICE)

        # Sets same weights from critics to critics target
        self.challenge_predict_critics_target.load_state_dict(
            self.challenge_predict_critics.state_dict()
        )

        # Set critic optimizer
        self.challenge_predict_critics_optimizer = torch.optim.Adam(
            self.challenge_predict_critics.parameters(), lr=LR
        )

        ### MAIN ACTORS
        self.actorList = []
        self.loadModelsList = ["Op3", "DOWN", "SMAA2", "STORE2", "Huller3"]

        for x in range(CHALLENGE_AMOUNT):
            # Actor Setup OP
            self.actor = Actor(self.state_size, self.action_size, self.max_action_t).to(
                DEVICE
            )
            self.actor.load_state_dict(
                torch.load(
                    "TD3/loadModels/bipedalTD3{}.pth".format(self.loadModelsList[x])
                )
            )
            self.actorList.append(self.actor)

    def get_predict_actor_action(self, lidar_data):
        # gir den her linje nogen mening?
        lidar_data = torch.tensor(lidar_data.reshape(1, -1)).float().to(DEVICE)
        actor_data = (
            self.challenge_predict_actor(lidar_data).cpu().data.numpy().flatten()
        )

        return actor_data

    def select_action(self, state, current_actor, noise=0.1):
        # Get best action based on current policy
        state = torch.tensor(state.reshape(1, -1)).float().to(DEVICE)
        action = self.actorList[current_actor](state).cpu().data.numpy().flatten()

        return action

    def train(self):
        reward_history = []
        current_actor_history = [0] * CHALLENGE_AMOUNT
        episodes_done = 0
        state = self.env.reset()
        done = False
        episode_reward = 0
        episode_steps = 0
        peaks_counter = 0
        lidar_data = state[-10:]
        print(f"Observation space:\n{state}")
        print(f"Last 10:\n{lidar_data}")

        print(type(state))
        print(type(lidar_data))

        print(state.shape)
        print(lidar_data.shape)

        print(state.reshape(1, -1).shape)
        print(lidar_data.reshape(1, -1).shape)

        for steps_done in range(TOTAL_STEPS):

            # NEW EXECUTION PLACE WEEEEEEEEEEEEEEEEE
            pick_best_actor_data = self.get_predict_actor_action(lidar_data)
            current_actor = pick_best_actor_data.argmax()
            current_actor_history[current_actor] += 1

            # Select action
            if self.warmup_steps > steps_done:
                action = self.env.action_space.sample()

            else:
                action = (
                    self.select_action(np.array(state), current_actor)
                    + np.random.normal(0, EXPLORE_NOISE, size=self.action_size)
                ).clip(-self.max_action, self.max_action)

            # Execute actions
            next_state, reward, done, info = self.env.step(action)

            # if reward <= -100:
            #    reward = -5

            # Append transition to memory
            self.memory.append(
                (
                    state,
                    action,
                    reward,
                    next_state,
                    done,
                    pick_best_actor_data,
                    lidar_data,
                )
            )

            # Update state, Update reward, Update episode steps
            state = next_state
            lidar_data = state[-10:]
            episode_reward += reward
            episode_steps += 1
            # save some data on our new component

            # Optimize if replay memory is large enough
            if len(self.memory) > MEMORY_SIZE_MIN:
                self.optimize(episode_steps, current_actor)

            if done:
                episodes_done += 1
                reward_history.append(episode_reward)
                print(
                    f"Episode: {episodes_done} --- CA: {current_actor} --- Reward: {episode_reward} --- ES/TS: {episode_steps} / {steps_done}"
                )
                state = self.env.reset()
                lidar_data = state[-10:]
                done = False
                episode_reward = 0
                episode_steps = 0

        self.env.close()

        # Save Predict Network
        torch.save(
            self.challenge_predict_actor.state_dict(),
            "OwnTake/loadModels/bipedalOwnTake07/Predicter/challengePredicter01.pth",
        )

        ##FIGURE 1
        # Create Graph
        plt.figure(1)

        # Plot the Reward History
        plt.plot(reward_history)
        plt.title("Reward History")
        plt.xlabel("Episodes")
        plt.ylabel("Reward")

        # Save Graph
        plt.savefig("OwnTake/graphs/RewardGraph{}.png".format(VERSION_CONTROL))

        ##FIGURE 2
        plt.figure(2)

        # Plot the Actor History
        plt.bar(x=[0, 1, 2, 3, 4], height=current_actor_history)
        plt.title("Current Actor History")
        plt.xlabel("Actors")
        plt.ylabel("Actor Picked amount")

        # Save Graph
        plt.savefig("OwnTake/graphs/ActorGraph{}.png".format(VERSION_CONTROL))

    def optimize(self, current_iteration, current_actor):

        state, action, reward, next_state, done, pick_best_actor_data, lidar_data = (
            self.memory.sample()
        )

        with torch.no_grad():
            # Select action according to policy and add clipped noise

            # Select action according to policy and add clipped noise for challenge predict actor
            pred_noise = (torch.randn_like(pick_best_actor_data) * POLICY_NOISE).clamp(
                -NOISE_CLIP, NOISE_CLIP
            )
            pred_next_action = (
                self.challenge_predict_actor_target(lidar_data) + pred_noise
            ).clamp(-torch.tensor(1).to(DEVICE), torch.tensor(1).to(DEVICE))

            # Compute the target Q value for challenge predict actor
            pred_target_Q1, pred_target_Q2 = self.challenge_predict_critics_target(
                lidar_data, pred_next_action
            )
            pred_target_Q = torch.min(pred_target_Q1, pred_target_Q2)

            # right dimensions
            done = done.unsqueeze(1)
            reward = reward.unsqueeze(1)

            # Update Q values
            pred_target_Q = reward + (1 - done) * DISCOUNT_FACTOR * pred_target_Q

        # Get current Q estimates
        pred_current_Q1, pred_current_Q2 = self.challenge_predict_critics(
            lidar_data, pick_best_actor_data
        )

        # Compute critic loss
        pred_critics_loss = F.mse_loss(pred_current_Q1, pred_target_Q) + F.mse_loss(
            pred_current_Q2, pred_target_Q
        )

        # Optimize the critic
        self.challenge_predict_critics_optimizer.zero_grad()
        pred_critics_loss.backward()
        self.challenge_predict_critics_optimizer.step()

        if current_iteration % POLICY_FREQ == 0:
            # Compute actor loss
            pred_actor_loss = -self.challenge_predict_critics.Q1(
                lidar_data, self.challenge_predict_actor(lidar_data)
            ).mean()

            # Optimize the actor
            self.challenge_predict_actor_optimizer.zero_grad()
            pred_actor_loss.backward()
            self.challenge_predict_actor_optimizer.step()

            ### Update the frozen target models
            # critics for challenge predict
            for param, target_param in zip(
                self.challenge_predict_critics.parameters(),
                self.challenge_predict_critics_target.parameters(),
            ):
                target_param.data.copy_(
                    TAU * param.data + (1 - TAU) * target_param.data
                )
            # actor for challenge predict
            for param, target_param in zip(
                self.challenge_predict_actor.parameters(),
                self.challenge_predict_actor_target.parameters(),
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
                print(f"Observation space:\n{state}")

            if TEST_BOOL:
                print(f"Episode {i+1} Reward: {episode_reward}")

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
