from __future__ import annotations
from abc import ABC
from typing import Tuple
from torch.functional import Tensor
from common.networks import NNBase, ResBlock, SharedResNetwork
import torch.nn as nn
import numpy as np
from trainers.ppo.networks import FilteredNN
import torch as T
class Connect4ResNet(NNBase):
    def __init__(self, name='connect_4_res_nn', checkpoint_directory='tmp',shape:tuple=None,n_actions=None):
        super().__init__(name, checkpoint_directory)
        filters = 128
        fc1_dims = 512
        n_blocks = 5
        observation_rows = shape[1]
        observation_cols = shape[2]
        self._blocks = nn.ModuleList([ResBlock(filters) for _ in range(n_blocks)])
        self._shared = nn.Sequential(
            nn.Conv2d(shape[0],filters,3,1,1),
            *self._blocks
        )
        self._pi_head = nn.Sequential(
            nn.Conv2d(filters,filters,3,1,1),
            nn.Flatten(),
            nn.Linear(observation_cols*observation_rows*filters,fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims,n_actions)
        )
        self._v_head = nn.Sequential(
            nn.Conv2d(filters,filters,3,1,1),
            nn.Flatten(),
            nn.Linear(observation_cols*observation_rows*filters,fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims,1),
            nn.Tanh()
        )
    
    def forward(self,state:Tensor)->tuple[Tensor,Tensor]:
        shared :Tensor = self._shared(state)
        pi: Tensor = self._pi_head(shared)
        v:Tensor = self._v_head(shared)
        probs = pi.softmax(dim=-1)
        return probs,v

class Connect4FilteredNetworkBase(FilteredNN,ABC):
    def __init__(self, name, checkpoint_directory):
        super().__init__(name, checkpoint_directory)
    
    def adjust_probs_for_invalid_moves(self, probs: Tensor, observation: Tensor) -> Tensor:
        a: np.ndarray = observation.numpy().copy()
        s_shape = observation.shape
        # We are checking if columns with 0 values in the last row for both players then reshaping to (?,n_actions)
        valid_moves = np.logical_and(
            a[:, 0, -1, :] == 0, a[:, 1, -1, :] == 0).reshape((s_shape[0], 7)).astype(np.float)
        valid_moves = T.tensor(valid_moves,dtype=T.float32)
        probs = probs * valid_moves
        sum_v = T.sum(probs,dim=-1,keepdim=True)
        probs = probs / sum_v
        return probs
    
    def forward(self,state:Tensor)->Tuple[Tensor,Tensor]:
        shared: Tensor = self._shared(state)
        pi: Tensor = self._pi_head(shared)
        v: Tensor = self._v_head(shared)
        probs = pi.softmax(dim=-1)
        # filter invalid moves
        probs = self.adjust_probs_for_invalid_moves(probs,state)
        return probs , v

class Connect4FilteredSharedResNN(SharedResNetwork,Connect4FilteredNetworkBase):
    def __init__(self, name='shared_res_network', checkpoint_directory='tmp', shape: tuple = None, n_actions: int = None, filters=128, fc_dims=512, n_blocks=5):
        super().__init__(name=name, checkpoint_directory=checkpoint_directory, shape=shape, n_actions=n_actions, filters=filters, fc_dims=fc_dims, n_blocks=n_blocks)
    
    def forward(self,state:Tensor)->Tuple[Tensor,Tensor]:
        shared: Tensor = self._shared(state)
        pi: Tensor = self._pi_head(shared)
        v: Tensor = self._v_head(shared)
        probs = pi.softmax(dim=-1)
        # filter invalid moves
        probs = self.adjust_probs_for_invalid_moves(probs,state)
        return probs , v
        