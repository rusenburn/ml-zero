from torch.functional import Tensor
from common.networks import NNBase
from abc import ABC,abstractmethod

class FilteredNN(NNBase,ABC):
    def __init__(self, name, checkpoint_directory):
        super().__init__(name, checkpoint_directory)

    def adjust_probs_for_invalid_moves(self,pi:Tensor,observation:Tensor)->Tensor:
        pass