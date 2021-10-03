from .networks import SmallConvNetwork , SharedResNetwork
from common.network_wrappers import TorchTrainableNNWrapper
from common.game import Game


class SmallConvWrapper(TorchTrainableNNWrapper):
    def __init__(self, game: Game, lr=2.5e-4, n_iterations=float('inf'), min_lr=1e-5) -> None:
        self.nn = SmallConvNetwork(
            shape=game.observation_shape, n_actions=game.n_actions)
        super().__init__(self.nn, lr, n_epochs=4, n_iterations=n_iterations, min_lr=min_lr)


class ResNetWrapper(TorchTrainableNNWrapper):
    def __init__(self, game: Game, lr=2.5e-4, n_iterations=float('inf'), min_lr=1e-5) -> None:
        self.nn = SharedResNetwork(
            shape=game.observation_shape, n_actions=game.n_actions)
        super().__init__(self.nn, lr=lr,n_epochs=10, n_iterations=n_iterations, min_lr=min_lr)
