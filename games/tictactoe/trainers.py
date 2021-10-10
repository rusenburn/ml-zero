from typing import Callable, List
from common.game import Game
from common.network_wrappers import TorchWrapper
from common.networks import NNBase, SharedResNetwork
from common.trainer import Trainer
from games.tictactoe.networks import FilteredSharedResNetwork
from trainers.pop3d.pop3d import POP3D
from trainers.ppo.pastselfPPO import PastSelfPPO
from trainers.ppo.ppo import PPO
from .game import TicTacToeGame
from .network_wrappers import ResNetWrapper


class TicTacToeTrainer(Trainer):
    def __init__(self, n_iterations, n_episodes, n_simulations, threshold, lr=2.5e-4, min_lr=1e-5) -> None:
        game = TicTacToeGame()
        nnet = SharedResNetwork(
            shape=game.observation_shape, n_actions=game.n_actions)
        super().__init__(game=game,
                         nn=nnet,
                         n_iterations=n_iterations,
                         n_episodes=n_episodes,
                         n_simulations=n_simulations,
                         threshold=threshold, n_epochs=10, batch_size=64, lr=lr, min_lr=min_lr)


class TicTacToePPOTrainer(PPO):
    def __init__(self, n_max_steps=100000, n_epochs=4, batch_size=20, n_batches=4, lr=0.00025, gamma=0.99, gae_lambda=0.95, policy_clip=0.1, testing_threshold=0.55, testing_intervals=50) -> None:
        game_fns = [lambda: TicTacToeGame() for x in range(1)]
        game = TicTacToeGame()
        nn = FilteredSharedResNetwork(shape=game.observation_shape, n_actions=game.n_actions,
                                      filters=128, fc_dims=512, n_blocks=3, filter_moves=False)
        super().__init__(game_fns, nn, n_max_steps=n_max_steps, n_epochs=n_epochs, batch_size=batch_size, n_batches=n_batches,
                         lr=lr, gamma=gamma, gae_lambda=gae_lambda, policy_clip=policy_clip, testing_threshold=testing_threshold, testing_intervals=testing_intervals)


class TicTacToePPOTrainer_(PPO):
    def __init__(self, n_max_steps=100000, n_epochs=4, batch_size=20, n_batches=4, lr=0.00025, gamma=0.99, gae_lambda=0.95, policy_clip=0.1, testing_threshold=0.55) -> None:
        game_fns = [lambda: TicTacToeGame() for x in range(1)]
        game = TicTacToeGame()
        nn = FilteredSharedResNetwork(shape=game.observation_shape, n_actions=game.n_actions,
                                      filters=128, fc_dims=512, n_blocks=3, filter_moves=True)
        super().__init__(game_fns, nn, n_max_steps=n_max_steps, n_epochs=n_epochs, batch_size=batch_size, n_batches=n_batches,
                         lr=lr, gamma=gamma, gae_lambda=gae_lambda, policy_clip=policy_clip, testing_threshold=testing_threshold)


class NewTrainer(POP3D):
    def __init__(self, n_max_steps=100000, n_epochs=4, batch_size=20, n_batches=4, lr=0.00025, gamma=0.99, gae_lambda=0.95, beta=10, testing_threshold=0.55, testing_intervals=50, min_lr=0.00001) -> None:
        game_fns = [lambda:TicTacToeGame() for _ in range(1)]
        game = TicTacToeGame()
        nn = FilteredSharedResNetwork(shape=game.observation_shape,n_actions=game.n_actions)
        self.wrapper = TorchWrapper(nn)
        super().__init__(game_fns, nn, n_max_steps=n_max_steps, n_epochs=n_epochs, batch_size=batch_size, n_batches=n_batches, lr=lr, gamma=gamma,
                         gae_lambda=gae_lambda, beta=beta, testing_threshold=testing_threshold, testing_intervals=testing_intervals, min_lr=min_lr)

class PastSelfTrainer(PastSelfPPO):
    def __init__(self, n_max_steps=100000, n_epochs=4, batch_size=20, n_batches=4, lr=0.00025, gamma=0.99, gae_lambda=0.95, policy_clip=0.1, testing_threshold=0.55, testing_intervals=50, min_lr=0.00001) -> None:
        game_fns = [lambda:TicTacToeGame() for _ in range(16)]
        game = TicTacToeGame()
        nn = FilteredSharedResNetwork(shape=game.observation_shape,n_actions=game.n_actions)
        super().__init__(game_fns, nn, n_max_steps=n_max_steps, n_epochs=n_epochs, batch_size=batch_size, n_batches=n_batches, lr=lr, gamma=gamma, gae_lambda=gae_lambda, policy_clip=policy_clip, testing_threshold=testing_threshold, testing_intervals=testing_intervals, min_lr=min_lr)
