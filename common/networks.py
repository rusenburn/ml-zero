from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Tuple
import torch as T
from torch.functional import Tensor
import torch.nn as nn
import os

from common.utils import get_device


class NNBase(nn.Module, ABC):
    def __init__(self, name, checkpoint_directory):
        super().__init__()
        self._file_name: str = os.path.join(checkpoint_directory, name)

    @abstractmethod
    def forward(self, state) -> Tensor:
        pass

    def save_model(self, path=None):
        try:
            if path:
                path = path
            else:
                path = self._file_name
            T.save(self.state_dict(), path)
            print(f'The nn was saved to {path}')
        except:
            print(f'could not save nn to {path}')

    def load_model(self, path=None):
        try:
            if path:
                path = path
            else:
                path = self._file_name
            self.load_state_dict(T.load(path))
            print(f'The nn was loaded from {path}')
        except:
            print(f'could not load nn from {path}')

class SharedResNetwork(NNBase):
    def __init__(self,
                 name='shared_res_network',
                 checkpoint_directory='tmp',
                 shape: tuple = None,
                 n_actions: int = None,
                 filters=128,
                 fc_dims=512,
                 n_blocks=3):
        super().__init__(name, checkpoint_directory)
        self._blocks = nn.ModuleList(
            [ResBlock(filters) for _ in range(n_blocks)])
        self._shared = nn.Sequential(
            nn.Conv2d(shape[0], filters, 3, 1, 1),
            *self._blocks
        )

        self._pi_head = nn.Sequential(
            nn.Conv2d(filters, filters, 3, 1, 1),
            nn.Flatten(),
            nn.Linear(shape[1]*shape[2]*filters, fc_dims),
            nn.ReLU(),
            nn.Linear(fc_dims, n_actions)
        )

        self._v_head = nn.Sequential(
            nn.Conv2d(filters, filters, 3, 1, 1),
            nn.Flatten(),
            nn.Linear(shape[1]*shape[2]*filters, fc_dims),
            nn.ReLU(),
            nn.Linear(fc_dims, 1),
            nn.Tanh()
        )
        device = get_device()
        self._blocks.to(device)
        self._shared.to(device)
        self._pi_head.to(device)
        self._v_head.to(device)

    def forward(self, state: Tensor) -> Tuple[Tensor, Tensor]:
        shared: Tensor = self._shared(state)
        pi: Tensor = self._pi_head(shared)
        v: Tensor = self._v_head(shared)
        probs: Tensor = pi.softmax(dim=-1)
        return probs, v

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self._block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.Conv2d(channels, channels, 3, 1, 1),
        )
        self._se = SqueezeAndExcite(channels, squeeze_rate=4)

    def forward(self, state: Tensor) -> Tensor:
        initial = state
        output: Tensor = self._block(state)
        output = self._se(output, initial)
        output += initial
        output = output.relu()
        return output

class SqueezeAndExcite(nn.Module):
    def __init__(self, channels, squeeze_rate):
        super().__init__()
        self.channels = channels
        self.prepare = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
        )
        self._fcs = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, int(channels//squeeze_rate)),
            nn.ReLU(),
            nn.Linear(int(channels//squeeze_rate), channels*2)
        )

    def forward(self, state: Tensor, input_: Tensor) -> Tensor:
        shape_ = input_.shape
        prepared: Tensor = self.prepare(state)
        prepared: Tensor = self._fcs(prepared)
        splitted = prepared.split(self.channels, dim=1)
        w = splitted[0]
        b = splitted[1]
        z = w.sigmoid()
        z = z.unsqueeze(-1).unsqueeze(-1).expand((-1, -
                                                  1, shape_[-2], shape_[-1]))
        b = b.unsqueeze(-1).unsqueeze(-1).expand((-1, -
                                                  1, shape_[-2], shape_[-1]))
        output = (input_*z) + b
        return output