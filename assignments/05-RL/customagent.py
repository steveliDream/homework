import gymnasium as gym
import random
import numpy as np


class Agent:
    """
    A simple Q learning.
    """

    def __init__(
        self, action_space: gym.spaces.Discrete, observation_space: gym.spaces.Box
    ):
        self.action_space = action_space
        self.observation_space = observation_space
        self.q_table = np.zeros((self.observation_space.shape[0], self.action_space.n))
        self.last_observation = None
        self.last_action = None

        # Hyperparameters
        self.alpha = 0.1
        self.gamma = 0.9
        self.epsilon = 0.1

    def act(self, observation: gym.spaces.Box) -> gym.spaces.Discrete:
        """
        as per requirement.
        """
        # Epsilon-greedy policy
        if random.uniform(0, 1) < self.epsilon:
            return self.action_space.sample()
        # else:
        #     q_values = self.q_table[observation.astype(np.int32)]
        #     action = np.argmax(q_values, axis=1)
        # action = action[-1]

        # self.last_observation = observation
        # self.last_action = action
        # return action

    def learn(
        self,
        observation: gym.spaces.Box,
        reward: float,
        terminated: bool,
        truncated: bool,
    ) -> None:
        """
        as per requirement
        """
        # Update Q-value
        if terminated:
            target_q = reward
        else:
            next_q_values = self.q_table[observation.astype(np.int32)]
            target_q = reward + self.gamma * np.max(next_q_values)

        current_q = self.q_table[
            self.last_observation.astype(np.int32), self.last_action
        ]
        td_error = target_q - current_q
        self.q_table[self.last_observation.astype(np.int32), self.last_action] += (
            self.alpha * td_error
        )

        # Decay exploration rate
        self.epsilon *= 0.95
