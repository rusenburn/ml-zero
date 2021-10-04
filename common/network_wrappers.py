from abc import ABC, abstractmethod
import torch as T
from torch.functional import Tensor
from common.game import Game, State
from common.networks import NNBase, SharedResNetworkTest
from typing import List, Tuple
import numpy as np
import os
from torch.nn.utils.clip_grad import clip_grad_norm_

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
        log_probs: Tensor
        v: Tensor
        self.nn.eval()
        observation_t = T.tensor([observation], dtype=T.float32)
        with T.no_grad():
            log_probs, v = self.nn(observation_t)

        probs = T.exp(log_probs).numpy()[0]
        return probs, v.data.cpu().numpy()[0]

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
                states = List[State]
                states, pis, vs = list(zip(*[examples[i] for i in sample_ids]))
                obs = [i.to_obs() for i in states]
                obs_t = T.tensor(obs, dtype=T.float32)
                target_pis = T.tensor(pis, dtype=T.float32)
                target_vs = T.tensor(vs, dtype=T.float32)

                log_probs: Tensor
                log_probs, out_v = self.nn(obs_t)
                out_probs = log_probs.exp()
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
    


class TestResNetWrapper(TorchTrainableNNWrapper):
    def __init__(self, game:Game,lr=0.00025, n_iterations=float('inf'),min_lr=1e-5) -> None:
        self.nn = SharedResNetworkTest(
            shape=game.observation_shape,n_actions=game.n_actions)
        super().__init__(self.nn, lr=lr, n_epochs=10, batch_size=64, n_iterations=n_iterations, min_lr=min_lr)