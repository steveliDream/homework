import gymnasium as gym
import numpy as np
import random


class Agent:
    """
    A simple Q learning.
    """

    def __init__(
        self, action_space: gym.spaces.Discrete, observation_space: gym.spaces.Box
    ):
        self.action_space = action_space
        self.observation_space = observation_space
        self.bin_size = 10
        self.bins = np.array(
            [
                np.linspace(
                    observation_space.low[i], observation_space.high[i], self.bin_size
                )
                for i in range(self.observation_space.shape[0])
            ]
        )
        sizes = [self.bin_size + 1 for _ in range(self.observation_space.shape[0])] + [
            self.action_space.n
        ]
        self.q_table = np.zeros(sizes)

        # Hyperparameters
        self.alpha = 0.1
        self.gamma = 0.99
        self.epsilon = 0.1

    def grid(self, observation):
        return tuple(np.digitize(obs, bin) for obs, bin in zip(observation, self.bins))

    def act(self, observation: gym.spaces.Box) -> gym.spaces.Discrete:
        """
        as per requirement.
        """
        # Epsilon-greedy policy
        if random.uniform(0, 1) < self.epsilon:
            return self.action_space.sample()
        return np.argmax(self.q_table[self.grid(observation)])

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
        grid_obs = self.grid(observation)
        if terminated or truncated:
            self.q_table[grid_obs] = reward
        else:
            max_future_q = np.max(self.q_table[grid_obs])
            current_q = self.q_table[grid_obs]
            new_q = (1 - self.alpha) * current_q + self.alpha * (
                reward + self.gamma * max_future_q
            )
            self.q_table[grid_obs] = new_q

        # Decay exploration rate
        self.epsilon *= 0.99
