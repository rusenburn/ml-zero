
from typing import List, Tuple

from torch.functional import Tensor
from torch.optim.lr_scheduler import _LRScheduler
from common.network_wrappers import NNWrapper
import numpy as np
import torch as T
from torch.distributions import Categorical
from torch.nn.utils.clip_grad import clip_grad_norm_
from common.networks import NNBase
import os

from common.utils import get_device


class PPONNWrapper(NNWrapper):
    def __init__(self, nn: NNBase, lr, policy_clip, n_iters, epochs, n_workers, worker_steps, batch_size, gamma, gae_lambda, observation_shape, min_lr) -> None:
        super().__init__()
        self.epochs = epochs
        self.nn = nn
        self.n_workers: int = n_workers
        self.worker_steps: int = worker_steps
        self.gamma: float = gamma
        self.gae_lambda: float = gae_lambda
        self.batch_size: int = batch_size
        self.optimizer = T.optim.Adam(self.nn.parameters(), lr=lr)
        self.policy_clip: float = policy_clip
        self.max_grad_norm = 0.5
        self.observation_shape: tuple = observation_shape

        min_lr_ratio = min_lr / lr if min_lr < lr else 1
        self.scheduler: _LRScheduler = T.optim.lr_scheduler.LambdaLR(
            optimizer=self.optimizer,
            lr_lambda=lambda step: max(1-step/n_iters, min_lr_ratio)
        )

    def predict(self, observation: np.ndarray) -> Tuple[int, float]:
        self.nn.eval()
        probs: Tensor
        v: Tensor
        observation_t = T.tensor([observation], dtype=T.float32,device=get_device())
        # TODO send to device
        # Done
        with T.no_grad():
            probs, v = self.nn(observation_t)
        return probs.data.cpu().numpy()[0], v.data.cpu().numpy()[0]

    def load_check_point(self, folder='tmp', file='nn'):
        path = None
        if folder and file:
            path = os.path.join(folder, file)
        self.nn.load_model(path)

    def save_check_point(self, folder='tmp', file='nn'):
        path = None
        if folder and file:
            path = os.path.join(folder, file)
        self.nn.save_model(path)

    def ppo_predict(self, observation: np.ndarray) -> Tuple[int, float, np.ndarray]:
        probs: Tensor
        value: Tensor
        self.nn.eval()
        observation_t = T.tensor([observation], dtype=T.float32,device=get_device())
        # TODO Move to device
        # Done
        probs, value = self.nn(observation_t)
        dist: Categorical = Categorical(probs)
        action_sample = dist.sample()
        log_probs = T.squeeze(dist.log_prob(action_sample)).item()
        action = T.squeeze(action_sample).item()
        value = T.squeeze(value).item()
        return action, value, log_probs

    def train(self, examples: List):
        self.nn.train()
        for _ in range(self.epochs):
            observations_batches, action_batches, log_probs_batches, value_batches, reward_batches, done_batches = examples
            advantages = self._calculate_advantages(
                reward_batches, value_batches, done_batches)

            batches = self._prepare_batches()

            states_arr, actions_arr, log_probs_arr, values_arr = self._reshape_batches(
                observations_batches, action_batches, log_probs_batches, value_batches)

            # TODO move values Tensor to device
            # Done
            values = T.tensor(values_arr.copy(),device=get_device())
            for batch in batches:
                # TODO move tensors to device
                # Done
                observations = T.tensor(states_arr[batch], dtype=T.float,device=get_device())
                old_logprobs = T.tensor(log_probs_arr[batch],device=get_device())
                actions = T.tensor(actions_arr[batch],device=get_device())
                ##
                probs: Tensor
                critic_value: Tensor
                probs, critic_value = self.nn(observations)
                dist: Categorical = Categorical(probs)

                entropy: Tensor = dist.entropy().mean()
                critic_value = critic_value.squeeze()
                new_logprobs: Tensor = dist.log_prob(actions)
                prob_ratio = (new_logprobs-old_logprobs).exp()
                weighted_probs = advantages[batch] * prob_ratio
                weighted_clipped_probs = T.clamp(
                    prob_ratio, 1-self.policy_clip, 1+self.policy_clip)
                weighted_clipped_probs *= advantages[batch]
                actor_loss = -T.min(weighted_probs,
                                    weighted_clipped_probs).mean()
                returns = advantages[batch] + values[batch]
                critic_loss = (returns-critic_value)**2
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

                self.optimizer.zero_grad()

                total_loss.backward()

                if self.max_grad_norm:
                    clip_grad_norm_(self.nn.parameters(),
                                    max_norm=self.max_grad_norm)
                self.optimizer.step()
        if self.scheduler:
            self.scheduler.step()

    def _calculate_advantages(self, rewards: np.ndarray, values: np.ndarray, dones: np.ndarray):
        advantages_arr = np.zeros(
            (self.n_workers, self.worker_steps), dtype=np.float32)
        for i in range(self.n_workers):
            for t in range(self.worker_steps-1):
                discount = 1
                a_t = 0
                #
                alter = -1
                #
                for k in range(t, self.worker_steps-1):
                    current_reward = rewards[i][k] * alter * -1
                    current_val = values[i][k]
                    next_val = values[i][k+1] * alter
                    next_val = next_val * alter

                    a_t += discount * (current_reward + self.gamma*next_val*(
                        1-int(dones[i][k])) - current_val)
                    discount *= self.gamma*self.gae_lambda
                    alter *= -1
                advantages_arr[i][t] = a_t
        # TODO move advantages to device
        # Done
        advantages = T.tensor(advantages_arr.flatten().copy(),device=get_device())
        return advantages

    def _prepare_batches(self):
        batch_start = np.arange(
            0, self.n_workers*self.worker_steps, self.batch_size)
        indices = np.arange(
            self.n_workers*self.worker_steps, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]
        return batches

    def _reshape_batches(self, observation_batches, action_batches, log_probs_batches, value_batches):
        observation_arr = observation_batches.reshape(
            (self.n_workers*self.worker_steps, *self.observation_shape))

        actions_arr = action_batches.reshape(
            (self.n_workers*self.worker_steps,))

        log_probs_arr = log_probs_batches.reshape(
            (self.n_workers*self.worker_steps,))

        values_arr = value_batches.reshape(
            (self.n_workers*self.worker_steps,))

        return observation_arr, actions_arr, log_probs_arr, values_arr
