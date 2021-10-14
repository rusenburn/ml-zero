from typing import Callable, List
from torch.optim.lr_scheduler import _LRScheduler
from common.game import Game, SyncGameVec
from common.network_wrappers import AIGame, TorchWrapper
from common.networks import NNBase
from common.utils import get_device
from trainers.ppo.ppo import PPO
import copy
import torch as T
import time
import numpy as np

class PastSelfPPO(PPO):
    def __init__(self, game_fns: List[Callable[[], Game]], nn: NNBase, n_max_steps=100000, n_epochs=4, batch_size=20, n_batches=4, lr=0.00025, gamma=0.99, gae_lambda=0.95, policy_clip=0.1, testing_threshold=0.55, testing_intervals=50, min_lr=0.00001) -> None:
        super().__init__(game_fns, nn, n_max_steps=n_max_steps, n_epochs=n_epochs, batch_size=batch_size, n_batches=n_batches, lr=lr, gamma=gamma,
                         gae_lambda=gae_lambda, policy_clip=policy_clip, testing_threshold=testing_threshold, testing_intervals=testing_intervals, min_lr=min_lr)
        self._instatiate_games(game_fns,nn)
        self._next_nn_idx = 0
        self.set_enemy_intervals = 50

    def train_wrapper(self, wrapper: TorchWrapper) -> TorchWrapper:
        # TODO fix wrapper thing
        self.optimizer = T.optim.Adam(self.wrapper.nn.parameters(), lr=self.lr)
        old_copy = copy.deepcopy(self.wrapper)
        old_state_dict = self.wrapper.nn.state_dict()
        self.wrapper.save_check_point('tmp', 'old')
        n_iters = int(self.n_max_steps // self.all_batches_size)
        scheduler: _LRScheduler = T.optim.lr_scheduler.LambdaLR(
            optimizer=self.optimizer,
            lr_lambda=lambda step: max(1-step/n_iters, self.min_lr_ratio))
        t_start = time.time()

        for i in range(n_iters):
            last_values = self._collect_training_examples()
            examples = self.memory.sample()
            self._train(examples,last_values)
            self.memory.reset()
            scheduler.step()
            if i and i % self.testing_intervals == 0:
                win_ratio = self._pit(self.wrapper, old_copy, 100)
                print(f'Win ratio vs old is {win_ratio:0.2f}.')
                if win_ratio < self.testing_threshold:
                    self.wrapper.nn.load_state_dict(old_state_dict)
                elif win_ratio >= self.testing_threshold:
                    old_state_dict = self.wrapper.nn.state_dict()
                    old_copy.nn.load_state_dict(old_state_dict)

                done_steps = self.all_batches_size * (i+1)
                delta_time = time.time() - t_start
                fps = done_steps // delta_time
                print(f'fps is {fps}')
            if i and i % self.set_enemy_intervals == 0:
                if len(self.copies):
                    self._load_next_from(self.wrapper)
        return self.wrapper

    def _instatiate_games(self, game_fns: List[Callable[[], Game]], nn: NNBase):
        self.copies = [copy.deepcopy(nn) for _ in range(len(game_fns)//2)]
        nns = [*self.copies]
        [nns.append(nn) for _ in range(len(game_fns) - len(self.copies))]
        games = [AIGame(fn(), n) for fn, n in zip(game_fns, nns)]
        assert len(nns) == len(game_fns) == len(games)
        self.vec_envs = SyncGameVec(games)

    def _load_next_from(self, wrapper: TorchWrapper):
        print(f'... Settings Current Policy as an Enemy ...')
        state_dict = wrapper.nn.state_dict()
        next_index = self._next_nn_idx % len(self.copies)
        next_copy = self.copies[next_index]
        next_copy.load_state_dict(state_dict)
        self._next_nn_idx +=1
    
    def _calculate_advantages(self, rewards: np.ndarray, values: np.ndarray, dones: np.ndarray,last_values:np.ndarray):
        adv_arr = np.zeros(
            (self.n_workers, self.worker_steps+1), dtype=np.float32)
        for i in range(self.n_workers):
            for t in reversed(range(self.worker_steps)):
                current_reward = rewards[i][t]
                current_val = values[i][t]
                if t == self.worker_steps -1 :
                    next_val = last_values[i]
                else:
                    next_val = values[i][t+1]
                delta = current_reward + (self.gamma * next_val * (1- int(dones[i][t]))) - current_val
                adv_arr[i][t] = delta + (self.gamma*self.gae_lambda*adv_arr[i][t+1] * (1-int(dones[i][t])))
        adv_arr = adv_arr[:,:-1]
        advantages = T.tensor(adv_arr.flatten().copy(),device=get_device())
        return advantages
