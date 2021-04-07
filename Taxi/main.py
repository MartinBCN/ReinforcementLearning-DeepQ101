from agent import Agent
from monitor import interact, plot
import gym
import numpy as np

env = gym.make('Taxi-v2')

state = env.reset()
action = env.action_space.sample()


observation, reward, done, info = env.step(action)  # take a random action

agent = Agent(n_action=env.action_space.n, alpha=0.1, gamma=0.8,
              epsilon_decay=0.9, epsilon_start=0.2, epsilon_min=0.0)
avg_rewards, best_avg_reward = interact(env, agent, num_episodes=25000)

plot(avg_rewards, best_avg_reward)
