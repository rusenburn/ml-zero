from common.networks import SharedResNetwork
from common.network_wrappers import TorchWrapper
from common.game import Game

class ResNetWrapper(TorchWrapper):
    def __init__(self, game: Game) -> None:
        nnet = SharedResNetwork(
            shape=game.observation_shape, n_actions=game.n_actions,filters=128,fc_dims=512,n_blocks=3)
        super().__init__(nnet)
