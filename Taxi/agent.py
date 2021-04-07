from typing import Union

import numpy as np
from collections import defaultdict


class Agent:

    def __init__(self, n_action: int = 6, epsilon_start: float = 0.9, epsilon_min: float = 0.1,
                 epsilon_decay: float = 0.9, alpha: float = 0.01, gamma: float = 1.0):
        """
        Parameters
        ----------
        n_action: int = 6
            number of actions available to the agent
        epsilon_start: float, default 0.9
        epsilon_min: float, default 0.1
        epsilon_decay: float, default 0.9
        alpha: float, default 0.01
        gamma: float, default 1.0
        """
        self.n_action = n_action
        self.Q = defaultdict(lambda: np.zeros(self.n_action))

        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.alpha = alpha
        self.gamma = gamma

    def select_action(self, state: Union[np.int, int]) -> int:
        """
        Given the state, select an action.

        Parameters
        ----------
        state: Union[np.int, int]

        Returns
        -------
        int:
            an integer, compatible with the task's action space
        """

        action = np.random.choice(np.arange(self.n_action), p=self.get_policy(state))

        return action

    def get_policy(self, state: Union[np.int, int]) -> np.array:

        policy = [self.epsilon / self.n_action] * self.n_action

        greedy_choice = int(np.argmax(self.Q[state]))
        policy[greedy_choice] += 1 - self.epsilon
        return np.array(policy)

    def get_best_action(self, state: Union[np.int, int]) -> int:
        return int(np.argmax(self.Q[state]))

    def step(self, state: Union[np.int, int], action: int, reward: int,
             next_state: Union[np.int, int], done: bool) -> None:
        """
        Update the agent's knowledge, using the most recently sampled tuple.

        Parameters
        ----------
        state: Union[np.int, int]
            the previous state of the environment
        action: int
            the agent's previous choice of action
        reward: int
            last reward received
        next_state: Union[np.int, int]
            the current state of the environment
        done: bool
            whether the episode is complete (True or False)

        Returns
        -------

        """

        # next_action = self.select_action(next_state)
        next_action = self.get_best_action(next_state)

        self.Q[state][action] += \
            self.alpha * (reward + self.gamma * self.Q[next_state][next_action] - self.Q[state][action])

        if done:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
