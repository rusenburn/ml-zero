from common.network_wrappers import TorchWrapper
from common.trainer import Trainer
from games.connect4.networks import Connect4FilteredSharedResNN
from trainers.pop3d.pop3d import POP3D
from trainers.ppo.ppo import PPO
from .network_wrappers import Connect4ResNetWrapper
from .game import ConnectFourGame


class Connect4Trainer(Trainer):
    def __init__(self, n_iterations, n_episodes, n_simulations, threshold, lr=2.5e-4, min_lr=1e-5) -> None:
        game = ConnectFourGame()
        trainable_nn = Connect4ResNetWrapper(
            game, lr=lr, n_iterations=n_iterations, min_lr=min_lr)
        super().__init__(game=game,
                         trainable_nn=trainable_nn,
                         n_iterations=n_iterations,
                         n_episodes=n_episodes,
                         n_simulations=n_simulations,
                         threshold=threshold)


class Connect4PPOTrainer(POP3D):
    def __init__(self, n_max_steps=100000, n_epochs=4, batch_size=20, n_batches=4, lr=0.00025, gamma=0.99, gae_lambda=0.95, policy_clip=0.1, testing_threshold=0.55, testing_intervals=50, min_lr=0.00001) -> None:
        game_fns = [lambda:ConnectFourGame() for _ in range(8)]
        game = ConnectFourGame()
        nnet = Connect4FilteredSharedResNN(
            shape=game.observation_shape, n_actions=game.n_actions)
        super().__init__(game_fns, nnet, n_max_steps=n_max_steps, n_epochs=n_epochs, batch_size=batch_size, n_batches=n_batches, lr=lr, gamma=gamma,
                         gae_lambda=gae_lambda, testing_threshold=testing_threshold, testing_intervals=testing_intervals, min_lr=min_lr)
