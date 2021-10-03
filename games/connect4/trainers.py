from common.trainer import Trainer
from .network_wrappers import Connect4ResNetWrapper
from .game import ConnectFourGame


class Connect4Trainer(Trainer):
    def __init__(self, n_iterations, n_episodes, n_simulations, threshold, lr=2.5e-4,min_lr=1e-5) -> None:
        game = ConnectFourGame()
        trainable_nn = Connect4ResNetWrapper(
            game, lr=lr, n_iterations=n_iterations, min_lr=min_lr)
        super().__init__(game=game,
                         trainable_nn=trainable_nn,
                         n_iterations=n_iterations,
                         n_episodes=n_episodes,
                         n_simulations=n_simulations,
                         threshold=threshold)
