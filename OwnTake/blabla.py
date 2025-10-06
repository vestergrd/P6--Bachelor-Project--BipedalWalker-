import gym

env = gym.make("BipedalWalker-v3", render_mode="human")
observation, info = env.reset(seed=42)
for i in range(1000):
    action = env.action_space.sample()  # this is where you would insert your policy
    observation, reward, terminated, truncated, info = env.step(action)
    print(f"Observation space:\n{observation}")

    if terminated or truncated:
        observation, info = env.reset()

env.close()
