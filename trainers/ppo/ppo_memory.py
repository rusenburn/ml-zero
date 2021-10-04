from __future__ import annotations
import numpy as np


class PPOMemory():
    def __init__(self, observation_shape, n_worker_steps, n_workers) -> None:
        self.observation_shape = observation_shape
        self.n_worker_steps = n_worker_steps
        self.n_workers = n_workers
        self.reset()

    def reset(self):
        self.current_worker = 0
        self.current_step = 0
        self.states = np.zeros((self.n_workers, self.n_worker_steps,*self.observation_shape), dtype=np.float32)
        self.actions = np.zeros(
            (self.n_workers, self.n_worker_steps), dtype=np.int32
        )
        self.rewards = np.zeros(
            (self.n_workers, self.n_worker_steps), dtype=np.float32
        )
        self.log_probs = np.zeros(
            (self.n_workers, self.n_worker_steps), dtype=np.float32
        )
        self.values = np.zeros(
            (self.n_workers, self.n_worker_steps), dtype=np.float32
        )
        self.dones = np.zeros(
            (self.n_workers, self.n_worker_steps), dtype=np.bool_)

    def remember(self, state: np.ndarray, action: int, log_probs: float, value: float, reward: float, done: bool):
        self.states[self.current_worker, self.current_step] = np.array(state)
        self.actions[self.current_worker, self.current_step] = action
        self.rewards[self.current_worker, self.current_step] = reward
        self.log_probs[self.current_worker, self.current_step] = log_probs
        self.values[self.current_worker, self.current_step] = value
        self.dones[self.current_worker, self.current_step] = done
        self._incerement_indices()

    def sample(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return self.states, self.actions, self.log_probs, self.values, self.rewards, self.dones

    def _incerement_indices(self):
        if self.current_worker == self.n_workers-1:
            self.current_worker = 0
            if self.current_step == self.n_worker_steps-1:
                self.current_step = 0
            else:
                self.current_step += 1
        else:
            self.current_worker += 1
