from abc import ABC
from typing import Tuple
from torch.functional import Tensor
from common.networks import SharedResNetwork
import numpy as np
from trainers.ppo.networks import FilteredNN
import torch as T

class TicTacToeFilteredNetworkBase(FilteredNN, ABC):
    def __init__(self, name, checkpoint_directory):
        super().__init__(name, checkpoint_directory)

    def adjust_probs_for_invalid_moves(self, probs: Tensor, state: Tensor) -> Tensor:
        a = state.numpy().copy()
        s_shape = state.shape
        valid_moves = np.logical_and(a[:, 0, :, :] == 0, a[:, 1, :, :] == 0).reshape(
            (s_shape[0], 9)).astype(np.float)
        valid_moves = T.tensor(valid_moves,dtype=T.float32)
        prob_1 = probs * valid_moves
        sum_v = T.sum(prob_1,dim=-1,keepdim=True)
        prob_1 = prob_1 / sum_v
        return probs
        

class FilteredSharedResNetwork(SharedResNetwork,TicTacToeFilteredNetworkBase):
    def __init__(self,
                 name='shared_res_network',
                 checkpoint_directory='tmp',
                 shape=None,
                 n_actions=None,
                 filters=128,
                 fc_dims=512,
                 n_blocks=3,
                 filter_moves=True):
        super().__init__(name,checkpoint_directory,shape,n_actions,filters,fc_dims,n_blocks)
        self.filter_moves = filter_moves

    def forward(self, state: Tensor) -> Tuple[Tensor, Tensor]:
        shared: Tensor = self._shared(state)
        pi: Tensor = self._pi_head(shared)
        v: Tensor = self._v_head(shared)
        probs: Tensor = pi.softmax(dim=-1)
        if self.filter_moves:
            probs = self.adjust_probs_for_invalid_moves(probs,state)
        return probs, v