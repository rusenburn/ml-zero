from typing import Callable, List
from common.game import Game
from common.networks import NNBase
from common.trainer import Trainer
from common.network_wrappers import TestResNetWrapper
from games.tictactoe.networks import FilteredSharedResNetwork
from trainers.ppo.ppo import PPO
from .game import TicTacToeGame
from .network_wrappers import ResNetWrapper


class TicTacToeTrainer(Trainer):
    def __init__(self, n_iterations, n_episodes, n_simulations, threshold, lr=2.5e-4, min_lr=1e-5) -> None:
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
    def __init__(self, n_iterations, n_episodes, n_simulations, threshold, lr=2.5e-4, min_lr=1e-5) -> None:
        game = TicTacToeGame()
        trainable_nn = TestResNetWrapper(
            game, lr=lr, n_iterations=n_iterations, min_lr=min_lr)
        super().__init__(game, trainable_nn, n_iterations,
                         n_episodes, n_simulations, threshold)


class TicTacToePPOTrainer(PPO):
    def __init__(self, n_max_steps=100000, n_epochs=4, batch_size=20, n_batches=4, lr=0.00025, gamma=0.99, gae_lambda=0.95, policy_clip=0.1, testing_threshold=0.55) -> None:
        game_fns = [lambda: TicTacToeGame() for x in range(1)]
        game = TicTacToeGame()
        nn = FilteredSharedResNetwork(shape=game.observation_shape, n_actions=game.n_actions,
                                      filters=128, fc1_dims=512, n_blocks=3, filter_moves=True)
        super().__init__(game_fns, nn, n_max_steps=n_max_steps, n_epochs=n_epochs, batch_size=batch_size, n_batches=n_batches,
                         lr=lr, gamma=gamma, gae_lambda=gae_lambda, policy_clip=policy_clip, testing_threshold=testing_threshold)
