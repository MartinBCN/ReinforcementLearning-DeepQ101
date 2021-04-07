# 1) Set decaying epsilon
# 2) Allow constant alpha
# 3) Update Q
# 4) Choose action from Q
# 5) Discount factor

import sys
from typing import Tuple, List

import gym
import numpy as np
from collections import defaultdict
from plot_utils import plot_blackjack_values, plot_policy


class BlackjackAgent:

    def __init__(self, alpha: float = 1.0, gamma: float = 1.0, epsilon_start: float = 0.9, epsilon_min: float = 0.1,
                 epsilon_decay: float = 0.9):
        self.env = gym.make('Introduction-v0')
        # initialize empty dictionary of arrays
        self.Q = defaultdict(lambda: np.zeros(self.env.action_space.n))
        self.N = defaultdict(lambda: np.zeros(self.env.action_space.n))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

    def action_from_q(self, state: Tuple[int, int, bool]) -> int:
        """
        Get an epsilon-greedy action from the Q table

        Parameters
        ----------
        state: Tuple[int, int, bool]

        Returns
        -------
        int
            Action from the action space (here 0/1)
        """

        probabilities = [self.epsilon / 2, self.epsilon / 2]

        greedy_choice = int(np.argmax(self.Q[state]))
        probabilities[greedy_choice] = 1 - self.epsilon + self.epsilon / 2
        action = np.random.choice(np.arange(2), p=probabilities)
        return action

    def update_q(self, episode: List[Tuple[Tuple[int, int, bool], int, float]]) -> None:
        """ updates the action-value function estimate using the most recent episode """
        states, actions, rewards = zip(*episode)
        # prepare for discounting
        discounts = np.array([self.gamma ** i for i in range(len(rewards) + 1)])
        for i, state in enumerate(states):
            old_Q = self.Q[state][actions[i]]
            self.Q[state][actions[i]] = old_Q + self.alpha * (sum(rewards[i:] * discounts[:-(1 + i)]) - old_Q)

    def generate_episode(self) -> List[Tuple[Tuple[int, int, bool], int, float]]:
        episode = []
        state = self.env.reset()
        while True:
            action = self.action_from_q(state)
            next_state, reward, done, info = self.env.step(action)
            episode.append((state, action, reward))
            state = next_state
            if done:
                break

        return episode

    def mc_control(self, num_episodes: int) -> None:
        self.env.reset()

        # loop over episodes
        for i_episode in range(1, num_episodes + 1):
            # monitor progress
            if i_episode % 1000 == 0:
                print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
                sys.stdout.flush()

            episode = self.generate_episode()

            self.update_q(episode)

            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_policy(self) -> dict:
        return dict((k, np.argmax(v)) for k, v in self.Q.items())

    def get_q(self):
        return self.Q


if __name__ == '__main__':
    bja = BlackjackAgent(alpha=0.02)
    bja.mc_control(500000)

    policy = bja.get_policy()

    plot_policy(policy)