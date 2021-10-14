from common.network_wrappers import TorchWrapper
from common.game import Game
from .networks import Connect4ResNet

class Connect4ResNetWrapper(TorchWrapper):
    def __init__(self, game:Game,lr=0.00025,n_iterations=float('inf'),min_lr=1e-5) -> None:
        nnet = Connect4ResNet(shape=game.observation_shape,n_actions=game.n_actions)
        super().__init__(nnet,lr=lr,n_epochs=10,n_iterations=n_iterations,min_lr=min_lr)
