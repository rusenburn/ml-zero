from typing import Tuple
from torch.functional import Tensor
from common.networks import NNBase, ResBlock
import torch.nn as nn


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
