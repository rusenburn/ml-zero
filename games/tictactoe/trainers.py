from common.trainer import Trainer
from common.network_wrappers import TestResNetWrapper
from .game import TicTacToeGame
from .network_wrappers import ResNetWrapper

class TicTacToeTrainer(Trainer):
    def __init__(self, n_iterations, n_episodes, n_simulations, threshold, lr=2.5e-4,min_lr=1e-5) -> None:
        game = TicTacToeGame()
        trainable_nn = ResNetWrapper(
            game, lr=lr, n_iterations=n_iterations, min_lr=min_lr)
        super().__init__(game=game,
                         trainable_nn=trainable_nn,
                         n_iterations=n_iterations,
                         n_episodes=n_episodes,
                         n_simulations=n_simulations,
                         threshold=threshold)

class GeluTrainerTest(Trainer):
    def __init__(self, n_iterations, n_episodes, n_simulations, threshold,lr=2.5e-4,min_lr=1e-5) -> None:
        game = TicTacToeGame()
        trainable_nn = TestResNetWrapper(game,lr=lr,n_iterations=n_iterations,min_lr=min_lr)
        super().__init__(game, trainable_nn, n_iterations, n_episodes, n_simulations, threshold)