from __future__ import annotations
from torch.functional import Tensor
from common.networks import NNBase, ResBlock
import torch.nn as nn

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
        log_probs = pi.log_softmax(dim=-1)
        return log_probs,v