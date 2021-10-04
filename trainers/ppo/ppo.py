import copy
from typing import Callable, List, Tuple
import time
from common.arena import Arena, NNPlayer
from common.game import Game, State, SyncGameVec
from common.networks import NNBase
from common.trainer import TrainerBase
import torch as T
import numpy as np
from trainers.ppo.network_wrappers import PPONNWrapper
from trainers.ppo.ppo_memory import PPOMemory


class PPO(TrainerBase):
    def __init__(self, game_fns: List[Callable[[], Game]], nn: NNBase, n_max_steps=1e5, n_epochs=4, batch_size=20, n_batches=4, lr=2.5e-4, gamma=0.99, gae_lambda=0.95, policy_clip=0.1, testing_threshold=0.55, testing_intervals=50) -> None:
        super().__init__()
        games = [fn() for fn in game_fns]
        self.vec_envs = SyncGameVec(games)
        self.lr = lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.n_max_steps = n_max_steps
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.n_batches = n_batches
        self.all_batches_size = batch_size * n_batches
        self.game = game_fns[0]()

        assert self.all_batches_size % self.vec_envs.n_envs == 0
        worker_steps = self.all_batches_size // self.vec_envs.n_envs
        n_workers = self.vec_envs.n_envs

        self.nn = PPONNWrapper(nn, lr, policy_clip, n_epochs, n_workers, worker_steps,
                               batch_size, gamma, gae_lambda, self.vec_envs.observation_shape)
        self.memory = PPOMemory(
            self.vec_envs.observation_shape, worker_steps, n_workers)
        self.testing_threshold = testing_threshold
        self.testing_intervals = testing_intervals
        self.states = self.vec_envs.reset()

    def train(self) -> PPONNWrapper:
        old_copy = copy.deepcopy(self.nn)
        self.nn.save_check_point('tmp', 'old')
        n_iters = int(self.n_max_steps // self.all_batches_size)
        t_start = time.time()
        for i in range(n_iters):
            self._collect_training_examples()
            examples = self.memory.sample()
            self.nn.train(examples)
            self.memory.reset()
            if i and i % self.testing_intervals == 0:
                win_ratio = self._pit(self.nn, old_copy, 100)
                print(f'Win ratio vs old is {win_ratio:0.2f}')
                if win_ratio < self.testing_threshold:
                    self.nn.load_check_point('tmp', 'old')
                elif win_ratio >= self.testing_threshold:
                    self.nn.save_check_point('tmp', 'old')
                    old_copy.load_check_point('tmp', 'old')
                done_steps = self.all_batches_size * (i+1)
                delta_time = time.time()-t_start
                fps = done_steps // delta_time
                print(f'fps is {fps}')
        return self.nn

    def _collect_training_examples(self):
        steps_per_env = self.all_batches_size // self.vec_envs.n_envs
        for i in range(steps_per_env):
            result = self._step(self.states)
            for j in range(len(result[0])):
                self.memory.remember(
                    result[0][j], result[1][j], result[2][j], result[3][j], result[4][j], result[5][j])
            self.states = result[6]

    def _step(self, states: List[State]) -> Tuple[List[np.ndarray], List[int], List[float], List[float], List[bool], List[State]]:
        observations, actions,  log_probs, values, rewards, dones = [], [], [], [], [], []
        for s, in zip(states):
            obs = s.to_obs()
            a, v, log_prob = self.nn.ppo_predict(s.to_obs())
            observations.append(obs)
            actions.append(a)
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
