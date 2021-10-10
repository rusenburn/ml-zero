from __future__ import annotations
from abc import ABC
from typing import List

from torch.functional import Tensor
from torch.nn.utils.clip_grad import clip_grad_norm_
from common.networks import NNBase
from common.nnmcts import NNMCTS
from common.utils import dotdict
from common.arena import Arena, NNMCTSPlayer
from common.game import Game, State
from common.network_wrappers import NNWrapper, TorchWrapper
import copy
import numpy as np
import torch as T


class TrainerBase(ABC):
    def __init__(self,) -> None:
        super().__init__()

    def train(self):
        pass


class Trainer(TrainerBase):
    def __init__(self, game: Game,nn:NNBase, n_iterations, n_episodes, n_simulations, threshold,lr,n_epochs,batch_size,min_lr) -> None:
        super().__init__()
        self._game = game
        self._n_iterations = n_iterations
        self._n_episodes = n_episodes
        self._n_simulations = n_simulations
        self._threshold = threshold
        self._n_epochs = n_epochs
        self._wrapper = TorchWrapper(nn)
        self._lr = lr
        self._min_lr_ratio = min_lr/lr if min_lr <= lr else 1
        self._batch_size = batch_size

    def train(self) -> TorchWrapper:
        return self.train_wrapper(self._wrapper)

    def train_wrapper(self,wrapper:TorchWrapper)->TorchWrapper:
        self._wrapper = wrapper
        self._optimizer = T.optim.Adam(wrapper.nn.parameters(),self._lr)
        scheduler = T.optim.lr_scheduler.LambdaLR(
            optimizer=self._optimizer,
            lr_lambda=lambda step: max(1-step/self._n_iterations, self._min_lr_ratio),
            verbose = True)
        old_copy = copy.deepcopy(self._wrapper)
        old_state_dict = self._wrapper.nn.state_dict()
        for i in range(self._n_iterations):
            examples = []
            for j in range(self._n_episodes):
                examples+=self._execute_episode()
            
            # self._wrapper.save_check_point('tmp','old')
            old_state_dict = self._wrapper.nn.state_dict()
            old_copy.nn.load_state_dict(old_state_dict)
            self._train(examples)
            scheduler.step()
            win_ratio = self._pit(self._wrapper,old_copy,100)
            print(f'win ratio is {win_ratio:0.2f}')
            if win_ratio < self._threshold:
                # self._wrapper.load_check_point('tmp', 'old')
                self._wrapper.nn.load_state_dict(old_state_dict)
        return self._wrapper
    
    def _train(self,examples):
        n_examples = len(examples)
        print(f'training from {n_examples} examples.')
        for epoch in range(self._n_epochs):
            print(f'Epoch ::: {epoch+1}')
            self._wrapper.nn.train()
            batch_count = int(len(examples)/self._batch_size)
            for _ in range(batch_count):
                sample_ids = np.random.randint(
                    len(examples), size=self._batch_size)
                states : List[State]
                states, pis, vs = list(zip(*[examples[i] for i in sample_ids]))
                obs = [i.to_obs() for i in states]
                obs_t = T.tensor(obs, dtype=T.float32)
                target_pis = T.tensor(pis, dtype=T.float32)
                target_vs = T.tensor(vs, dtype=T.float32)

                out_probs: Tensor
                out_probs, out_v = self._wrapper.nn(obs_t)
                l_pi = self._loss_pi(target_pis, out_probs)
                l_v = self._loss_v(target_vs, out_v)
                total_loss: Tensor = l_pi + l_v

                self._optimizer.zero_grad()
                total_loss.backward()
                clip_grad_norm_(self._wrapper.nn.parameters(), 0.5)
                self._optimizer.step()

    def _pit(self, new: NNWrapper, old: NNWrapper, n_games):
        args = dotdict({
            'cpuct': 1,
            'numMCTSSims': 25
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
        mcts = NNMCTS(self._game, self._wrapper, args)
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
    
    def _loss_pi(self, targets: Tensor, outputs: Tensor):
        return T.torch.sum((targets - outputs) ** 2) / targets.size()[0]

    def _loss_v(self, targets:Tensor, outputs:Tensor):
        return T.torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]
