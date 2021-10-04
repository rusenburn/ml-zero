from abc import ABC
from typing import Tuple
from torch.functional import Tensor
from common.networks import NNBase, ResBlock
import torch.nn as nn
import numpy as np
from trainers.ppo.networks import FilteredNN
import torch as T


class SmallConvNetwork(NNBase):
    def __init__(self, name='small_conv_shared_nn', checkpoint_directory='tmp', shape=None, n_actions=None):
        super().__init__(name, checkpoint_directory)
        filters = 16
        fc_dims = 256
        flattened_n_neurons = filters * (shape[1]-1) * (shape[2]-1)
        self._pi = nn.Sequential(
            nn.Conv2d(shape[0], filters, 2, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(flattened_n_neurons, fc_dims),
            nn.ReLU(),
            nn.Linear(fc_dims, n_actions),
        )
        self._v = nn.Sequential(
            nn.Conv2d(shape[0], filters, 2, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(flattened_n_neurons, fc_dims),
            nn.ReLU(),
            nn.Linear(fc_dims, 1),
            nn.Tanh()
        )
        # self._pi.to()
        # self._v.to()

    def forward(self, state: Tensor) -> Tuple[Tensor, Tensor]:
        pi: Tensor = self._pi(state)
        v: Tensor = self._v(state)
        log_probs = pi.log_softmax(dim=-1)
        return log_probs, v


class SharedResNetwork(NNBase):
    def __init__(self,
                 name='shared_res_network',
                 checkpoint_directory='tmp',
                 shape=None,
                 n_actions=None,
                 filters=128,
                 fc1_dims=512,
                 n_blocks=3):
        super().__init__(name, checkpoint_directory)
        self._blocks = nn.ModuleList(
            [ResBlock(filters) for _ in range(n_blocks)])
        self._shared = nn.Sequential(
            nn.Conv2d(shape[0], filters, 3, 1, 1),
            *self._blocks,
        )

        self._pi_head = nn.Sequential(
            nn.Conv2d(filters, filters, 3, 1, 1),
            nn.Flatten(),
            nn.Linear(shape[1]*shape[2]*filters, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, n_actions)
        )

        self._v_head = nn.Sequential(
            nn.Conv2d(filters, filters, 3, 1, 1),
            nn.Flatten(),
            nn.Linear(shape[1]*shape[2]*filters, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, 1),
            nn.Tanh()
        )
        # TODO send to device

    def forward(self, state: Tensor) -> Tuple[Tensor, Tensor]:
        shared: Tensor = self._shared(state)
        pi: Tensor = self._pi_head(shared)
        v: Tensor = self._v_head(shared)
        log_probs: Tensor = pi.log_softmax(dim=-1)
        return log_probs, v


class TicTacToeFilteredNetworkBase(FilteredNN, ABC):
    def __init__(self, name, checkpoint_directory):
        super().__init__(name, checkpoint_directory)

    def adjust_probs_for_invalid_moves(self, pi: Tensor, state: Tensor) -> Tensor:
        a = state.numpy().copy()
        s_shape = state.shape
        valid_moves = np.logical_and(a[:, 0, :, :] == 0, a[:, 1, :, :] == 0).reshape(
            (s_shape[0], 9)).astype(np.float)
        extremes = np.iinfo(np.int32)
        valid_moves[valid_moves < 1] = extremes.min
        valid_moves[valid_moves > 0] = extremes.max
        t_valid_moves = T.tensor(valid_moves, dtype=T.float32)
        min_pi = T.min(pi, t_valid_moves)
        return min_pi


class FilteredSharedResNetwork(TicTacToeFilteredNetworkBase):
    def __init__(self,
                 name='shared_res_network',
                 checkpoint_directory='tmp',
                 shape=None,
                 n_actions=None,
                 filters=128,
                 fc1_dims=512,
                 n_blocks=3,
                 filter_moves=True):
        super().__init__(name, checkpoint_directory)
        self._blocks = nn.ModuleList(
            [ResBlock(filters) for _ in range(n_blocks)])
        self._shared = nn.Sequential(
            nn.Conv2d(shape[0], filters, 3, 1, 1),
            *self._blocks,
        )

        self._pi_head = nn.Sequential(
            nn.Conv2d(filters, filters, 3, 1, 1),
            nn.Flatten(),
            nn.Linear(shape[1]*shape[2]*filters, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, n_actions)
        )

        self._v_head = nn.Sequential(
            nn.Conv2d(filters, filters, 3, 1, 1),
            nn.Flatten(),
            nn.Linear(shape[1]*shape[2]*filters, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, 1),
        )
        # TODO send to device
        self.filter_moves = filter_moves

    def forward(self, state: Tensor) -> Tuple[Tensor, Tensor]:
        shared: Tensor = self._shared(state)
        pi: Tensor = self._pi_head(shared)
        v: Tensor = self._v_head(shared)
        if self.filter_moves:
            pi: Tensor = self.adjust_probs_for_invalid_moves(pi, state)
        probs: Tensor = pi.softmax(dim=-1)
        return probs, v
