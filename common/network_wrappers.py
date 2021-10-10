from abc import ABC, abstractmethod
import torch as T
from torch.functional import Tensor
from common.game import Game, State
from common.networks import NNBase
from typing import List, Tuple
import numpy as np
import os
from torch.nn.utils.clip_grad import clip_grad_norm_

from common.utils import get_device

class NNWrapper(ABC):
    '''
    Wrappes a neural network, with basic functionalities,
    like predict , save and load checkpoints.
    '''
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def predict(self, observation:np.ndarray) -> Tuple[np.ndarray, float]:
        '''
        takes a numpy array observation and returns an action and the evaluation 
        of that observation.
        '''
        
    
    @abstractmethod
    def save_check_point(self, folder='tmp', file='nn')->None:
        '''
        Saves a checkpoint into a file.
        '''
    
    @abstractmethod
    def load_check_point(self, folder='tmp', file='nn')->None:
        '''
        Loads a checkpoint from a file.
        '''

class TrainableWrapper(NNWrapper,ABC):
    '''
    Wrappes a neural network, includes basic functionalities plus 
    the ability to be trained.
    '''
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def train(self, examples:List[Tuple[State,np.ndarray,float]]):
        '''
        Trains the wrapped neural network using examples arguments.
        '''

class TorchWrapper(NNWrapper):
    '''
    Wrapper that supports Pytorch NN
    can predict , load checkpoint
    and save checkpoints.
    '''
    def __init__(self,nn:NNBase) -> None:
        super().__init__()
        self.nn = nn
    def predict(self, observation) -> Tuple[np.ndarray, float]:
        probs: Tensor
        v: Tensor
        self.nn.eval()
        # TODO move to device
        # Done
        observation_t = T.tensor([observation], dtype=T.float32,device=get_device())
        with T.no_grad():
            probs, v = self.nn(observation_t)

        return probs.data.cpu().numpy()[0], v.data.cpu().numpy()[0]

    def load_check_point(self, folder='tmp', file='nn'):
        path = None
        if folder and file:
            path = os.path.join(folder, file)
        self.nn.load_model(path)

    def save_check_point(self, folder='tmp', file='nn'):
        path = None
        if folder and file:
            path = os.path.join(folder, file)
        self.nn.save_model(path)

class TorchTrainableNNWrapper(TorchWrapper,TrainableWrapper):
    '''
    Trainable wrapper that supports Pytorch NN
    can train , predict , load checkpoint
    and save checkpoints
    '''
    def __init__(self, nn: NNBase,
                 lr=2.5e-4,
                 n_epochs=10,
                 batch_size=64,
                 n_iterations=float('inf'),
                 min_lr=1e-5
                 ) -> None:

        super().__init__(nn=nn)
        self.optim = T.optim.Adam(self.nn.parameters(), lr=lr)
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        min_lr_ratio = min_lr/lr if min_lr < lr else 1
        self.scheduler = T.optim.lr_scheduler.LambdaLR(
            optimizer=self.optim,
            lr_lambda=lambda step: max(min_lr_ratio, 1-step/n_iterations),
            verbose=True
        )

    def train(self, examples):
        n_examples = len(examples)
        print(f'training from {n_examples} examples.')
        for epoch in range(self.n_epochs):
            print(f'Epoch ::: {epoch+1}')
            self.nn.train()
            batch_count = int(len(examples)/self.batch_size)
            for _ in range(batch_count):
                sample_ids = np.random.randint(
                    len(examples), size=self.batch_size)
                states : List[State]
                states, pis, vs = list(zip(*[examples[i] for i in sample_ids]))
                obs = [i.to_obs() for i in states]
                obs_t = T.tensor(obs, dtype=T.float32)
                target_pis = T.tensor(pis, dtype=T.float32)
                target_vs = T.tensor(vs, dtype=T.float32)

                out_probs: Tensor
                out_probs, out_v = self.nn(obs_t)
                # out_probs = log_probs.exp()
                l_pi = self._loss_pi(target_pis, out_probs)
                l_v = self._loss_v(target_vs, out_v)
                total_loss: Tensor = l_pi + l_v

                self.optim.zero_grad()
                total_loss.backward()
                clip_grad_norm_(self.nn.parameters(), 0.5)
                self.optim.step()
        self.scheduler.step()

    def _loss_pi(self, targets: Tensor, outputs: Tensor):
        return T.torch.sum((targets - outputs) ** 2) / targets.size()[0]

    def _loss_v(self, targets, outputs):
        return T.torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]

class AIGame(Game,NNWrapper):
    def __init__(self,game:Game,nn:NNBase) -> None:
        super().__init__()
        self._wrapper = TorchWrapper(nn)
        self._game = game
        self._starting_player = 0
    
    def n_actions(self) -> int:
        return self._game.n_actions
    
    def observation_shape(self) -> tuple:
        return self._game.observation_shape
    def reset(self) -> State:
        state = self._game.reset()
        if self._starting_player:
            a = self._choose_action(state)
            state,_,_,_ = self._game.step(a)
        self._starting_player = 1-self._starting_player
        return state
    
    def render(self) -> None:
        self._game.render()
    
    def predict(self, observation: np.ndarray) -> Tuple[np.ndarray, float]:
        return self._wrapper.predict(observation)
    
    def save_check_point(self, folder='tmp', file='nn') -> None:
        return self._wrapper.save_check_point(folder,file)
    
    def load_check_point(self, folder='tmp', file='nn') -> None:
        return self._wrapper.load_check_point(folder,file)
    def step(self, action:int) -> Tuple[State, float, bool, dict]:
        state,reward,done,info = self._game.step(action)
        if done:
            return state,reward,done,info
        action = self._choose_action(state)
        state_2 ,reward_2,done_2,info_2 = self._game.step(action)
        reward -= reward_2
        return state_2,reward,done_2,info_2
        
    def _choose_action(self, state: State) -> int:
        # TODO use NN player instead
        probs,_ = self._wrapper.predict(state.to_obs())
        legal_actions = state.get_legal_actions()
        probs = [probs[a] if a in legal_actions else 0 for a in range(len(probs))]
        probs_sum = float(sum(probs))
        norm_probs = [x/probs_sum for x in probs]
        action = np.random.choice(len(norm_probs),p=norm_probs)
        return action
        
