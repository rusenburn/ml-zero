import copy
import math
from typing import Callable, List, Tuple
import time
from torch.distributions.categorical import Categorical

from torch.functional import Tensor
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.optim.lr_scheduler import _LRScheduler
from common.arena import Arena, NNPlayer
from common.game import Game, State, SyncGameVec
from common.network_wrappers import TorchWrapper
from common.networks import NNBase
from common.trainer import TrainerBase
import torch as T
import numpy as np
from common.utils import get_device
# from trainers.ppo.network_wrappers import PPONNWrapper
from trainers.ppo.ppo_memory import PPOMemory


class PPO(TrainerBase):
    def __init__(self, game_fns: List[Callable[[], Game]], nn: NNBase, n_max_steps=1e5, n_epochs=4, batch_size=20, n_batches=4, lr=2.5e-4, gamma=0.99, gae_lambda=0.95, policy_clip=0.1, testing_threshold=0.55, testing_intervals=50, min_lr=1e-5) -> None:
        super().__init__()
        games = [fn() for fn in game_fns]
        self.vec_envs = SyncGameVec(games)
        self.lr = lr
        self.policy_clip = policy_clip
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.n_max_steps = n_max_steps
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.n_batches = n_batches
        self.all_batches_size = batch_size * n_batches
        self.game = game_fns[0]()

        self.min_lr_ratio = min_lr / lr if min_lr < lr else 1
        assert self.all_batches_size % self.vec_envs.n_envs == 0
        self.worker_steps = self.all_batches_size // self.vec_envs.n_envs
        self.n_workers = self.vec_envs.n_envs
        self.testing_threshold = testing_threshold
        self.testing_intervals = testing_intervals
        self.states = self.vec_envs.reset()
        self.max_grad_norm = 0.5
        self.wrapper: TorchWrapper = TorchWrapper(nn)

        self.memory = PPOMemory(
            self.game.observation_shape, self.worker_steps, self.n_workers)

    def train(self) -> TorchWrapper:
        return self.train_wrapper(self.wrapper)

    def train_wrapper(self, wrapper: TorchWrapper) -> TorchWrapper:
        self.wrapper = wrapper
        self.optimizer = T.optim.Adam(self.wrapper.nn.parameters(), lr=self.lr)
        old_copy = copy.deepcopy(self.wrapper)
        # self.wrapper.save_check_point('tmp', 'old')
        old_state_dict = self.wrapper.nn.state_dict()
        n_iters = int(self.n_max_steps // self.all_batches_size)
        scheduler: _LRScheduler = T.optim.lr_scheduler.LambdaLR(
            optimizer=self.optimizer,
            lr_lambda=lambda step: max(1-step/n_iters, self.min_lr_ratio))
        t_start = time.time()
        for i in range(n_iters):
            last_values = self._collect_training_examples()
            examples = self.memory.sample()
            self._train(examples)
            self.memory.reset()
            scheduler.step()
            if i and i % self.testing_intervals == 0:
                win_ratio = self._pit(self.wrapper, old_copy, 100)
                print(f'Win ratio vs old is {win_ratio:0.0f}.')
                if win_ratio < self.testing_threshold:
                    self.wrapper.nn.load_state_dict(old_state_dict)
                elif win_ratio >= self.testing_threshold:
                    old_state_dict = self.wrapper.nn.state_dict()
                    old_copy.nn.load_state_dict(old_state_dict)
                done_steps = self.all_batches_size * (i+1)
                delta_time = time.time() - t_start
                fps = done_steps // delta_time
                print(f'fps is {fps}')
        return self.wrapper

    def _collect_training_examples(self)->np.ndarray:
        steps_per_env = self.all_batches_size // self.vec_envs.n_envs
        for i in range(steps_per_env):
            result = self._step(self.states)
            for j in range(len(result[0])):
                self.memory.remember(
                    result[0][j], result[1][j], result[2][j], result[3][j], result[4][j], result[5][j])
            self.states:List[State] = result[6]
        # last_obs = [state.to_obs() for state in self.states]
        last_values = [self.wrapper.predict(state.to_obs())[1] for state in self.states]
        last_values = np.array(last_values,dtype=np.float32)
        return last_values

    def _step(self, states: List[State]) -> Tuple[List[np.ndarray], List[int], List[float], List[float], List[bool], List[State]]:
        observations, actions,  log_probs, values, rewards, dones = [], [], [], [], [], []
        for s, in zip(states):
            obs = s.to_obs()
            probs, v = self.wrapper.predict(s.to_obs())
            legal_actions = s.get_legal_actions()
            valid_probs = [
                probs[a] if a in legal_actions else 0 for a in range(len(probs))]
            probs_sum = float(sum(valid_probs))
            if probs_sum == 0:
                print('zero')
            norm_prob = [x/probs_sum for x in valid_probs]
            action = np.random.choice(len(norm_prob), p=norm_prob)
            prob = probs[action]
            log_prob = math.log(prob)
            observations.append(obs)
            actions.append(action)
            log_probs.append(log_prob)
            values.append(v)

        new_states, rewards, dones, _ = self.vec_envs.step(actions)
        return observations, actions, log_probs, values, rewards, dones, new_states

    def _pit(self, new, old, n_games):
        p_new = NNPlayer(new)
        p_old = NNPlayer(old)
        arena = Arena(p_new, p_old, self.game, n_games)
        fraction = arena.brawl()
        return fraction

    def _train(self, examples: List,last_values:np.ndarray):
        self.wrapper.nn.train()
        for _ in range(self.n_epochs):
            observations_batches, action_batches, log_probs_batches, value_batches, reward_batches, done_batches = examples
            advantages = self._calculate_advantages(
                reward_batches, value_batches, done_batches,last_values)

            batches = self._prepare_batches()

            states_arr, actions_arr, log_probs_arr, values_arr = self._reshape_batches(
                observations_batches, action_batches, log_probs_batches, value_batches)

            values = T.tensor(values_arr.copy(),device=get_device())
            for batch in batches:
                observations = T.tensor(states_arr[batch], dtype=T.float32,device=get_device())
                old_logprobs = T.tensor(log_probs_arr[batch], dtype=T.float32,device=get_device())
                actions = T.tensor(actions_arr[batch],device=get_device())
                ##
                probs: Tensor
                critic_value: Tensor
                probs, critic_value = self.wrapper.nn(observations)
                dist: Categorical = Categorical(probs)

                entropy: Tensor = dist.entropy().mean()
                critic_value = critic_value.squeeze()
                new_logprobs: Tensor = dist.log_prob(actions)
                prob_ratio = (new_logprobs-old_logprobs).exp()
                weighted_probs = advantages[batch] * prob_ratio
                weighted_clipped_probs = prob_ratio.clamp(
                    1-self.policy_clip, 1+self.policy_clip)
                weighted_clipped_probs *= advantages[batch]
                actor_loss = -T.min(weighted_probs,
                                    weighted_clipped_probs).mean()
                returns = advantages[batch] + values[batch]
                critic_loss = (returns - critic_value)**2
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

                self.optimizer.zero_grad()

                total_loss.backward()

                if self.max_grad_norm:
                    clip_grad_norm_(self.wrapper.nn.parameters(),
                                    max_norm=self.max_grad_norm)
                self.optimizer.step()

    def _calculate_advantages(self, rewards: np.ndarray, values: np.ndarray, dones: np.ndarray,last_values:np.ndarray):
        # For some reason it yields better result even though it is flawed
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
                    next_val = values[i][k+1]

                    a_t += discount * (current_reward + self.gamma*next_val*(
                        1-int(dones[i][k])) - current_val)
                    discount *= self.gamma*self.gae_lambda
                    alter *= -1
                advantages_arr[i][t] = a_t
        # returns = advantages_arr + values
        advantages = T.tensor(advantages_arr.flatten().copy(),device=get_device())
        return advantages

    def _prepare_batches(self):
        batch_start = np.arange(
            0, self.all_batches_size, self.batch_size)
        indices = np.arange(
            self.all_batches_size, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]
        return batches

    def _reshape_batches(self, observation_batches, action_batches, log_probs_batches, value_batches) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        observation_arr = observation_batches.reshape(
            (self.n_workers*self.worker_steps, *self.game.observation_shape))

        actions_arr = action_batches.reshape(
            (self.n_workers*self.worker_steps,))

        log_probs_arr = log_probs_batches.reshape(
            (self.n_workers*self.worker_steps,))

        values_arr = value_batches.reshape(
            (self.n_workers*self.worker_steps,))

        return observation_arr, actions_arr, log_probs_arr, values_arr

    
    def _calculate_advantages_improved(self, rewards: np.ndarray, values: np.ndarray, dones: np.ndarray,last_values:np.ndarray):
        adv_arr = np.zeros(
            (self.n_workers, self.worker_steps+1), dtype=np.float32)
        for i in range(self.n_workers):
            for t in reversed(range(self.worker_steps)):
                current_reward = rewards[i][t]
                current_val = values[i][t]
                if t == self.worker_steps-1: # if last step for worker 
                    next_val = last_values[i] * -1
                else:
                    next_val = values[i][t+1] * -1
                delta = current_reward + (self.gamma * next_val * (1- int(dones[i][t]))) - current_val
                adv_arr[i][t] = delta + (self.gamma*self.gae_lambda*adv_arr[i][t+1] * (1-int(dones[i][t]))) * -1
        adv_arr = adv_arr[:,:-1]
        returns_arr:np.ndarray = adv_arr + values
        adv_arr:np.ndarray = (adv_arr - adv_arr.mean())/adv_arr.std() + 1e-8

        advantages = T.tensor(adv_arr.flatten().copy(),device=get_device())
        returns = T.tensor(returns_arr.flatten().copy(),device=get_device())
        return advantages , returns
