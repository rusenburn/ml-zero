from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List
from games.tictactoe import TicTacToeGame
from common.nnmcts import NNMCTS
from common.utils import dotdict
from common.arena import Arena, NNMCTSPlayer
from common.game import Game, State
from common.network_wrappers import NNWrapper, TrainableWrapper
import copy
import numpy as np


class TrainerBase(ABC):
    def __init__(self,) -> None:
        super().__init__()

    def train(self):
        pass


class Trainer(TrainerBase):
    def __init__(self, game: Game, trainable_nn: TrainableWrapper, n_iterations, n_episodes, n_simulations, threshold) -> None:
        super().__init__()
        self._game = game
        self._nn = trainable_nn
        self._n_iterations = n_iterations
        self._n_episodes = n_episodes
        self._n_simulations = n_simulations
        self._threshold = threshold

    def train(self) -> TrainableWrapper:
        for i in range(self._n_iterations):
            examples = []
            for j in range(self._n_episodes):
                examples += self._execute_episode()

            self._nn.save_check_point('tmp', 'old')
            old_copy = copy.deepcopy(self._nn)
            self._nn.train(examples)

            win_ratio = self._pit(self._nn, old_copy, 100)
            print(f'win ratio is {win_ratio:0.3f}')
            if win_ratio < self._threshold:
                self._nn.load_check_point('tmp', 'old')
        return self._nn

    def _pit(self, new: NNWrapper, old: NNWrapper, n_games):
        args = dotdict({
            'cpuct': 1,
            'numMCTSSims': 5
        })
        p_new = NNMCTSPlayer(self._game, new, args)
        p_old = NNMCTSPlayer(self._game, old, args)
        arena = Arena(p_new, p_old, self._game, n_games=n_games)
        fraction = arena.brawl()
        return fraction

    def _execute_episode(self) -> List[State, np.ndarray, float]:
        args = dotdict({
            'cpuct': 1,
            'numMCTSSims': self._n_simulations
        })
        examples = []
        s = self._game.reset()
        mcts = NNMCTS(self._game, self._nn, args)
        current_player = 0
        while True:
            probs = mcts.probs(s)
            examples.append([s, probs, None, current_player])
            action = np.random.choice(len(probs), p=probs)
            s: State = s.move(action)
            current_player = 1-current_player
            if s.is_game_over():
                examples = self._assign_rewards(
                    examples, s.game_result(), current_player)
                return examples

    def _assign_rewards(self, examples, reward, last_player) -> tuple[State, np.ndarray, float]:
        for ex in examples:
            ex[2] = reward if ex[3] == last_player else -reward
        return [(x[0], x[1], x[2]) for x in examples]
