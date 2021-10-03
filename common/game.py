from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np


class State(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def get_legal_actions(self):
        '''
        Constructs a list of all
        possible actions from current state.
        Returns a list.
        '''
    @abstractmethod
    def is_game_over(self) -> bool:
        '''
        Returns True if the game is over
        or False if it is not
        '''
    @abstractmethod
    def game_result(self) -> int:
        '''
        Returns 1 if game is won
        -1 if game is lost or 0 
        if it drawn or not finished
        '''

    def move(self, action:int) -> State:
        '''
        Returns the new state of the game after
        peforming an action
        '''
    @abstractmethod
    def to_obs(self) -> np.ndarray:
        '''
        Converts the state into numpy array
        '''
    @abstractmethod
    def render(self) -> None:
        '''
        Renders the current state
        '''
    @abstractmethod
    def to_short(self) -> tuple:
        '''
        Returns short form for the current state
        can be used as a key
        '''


class Game(ABC):
    def __init__(self) -> None:
        super().__init__()

    @property
    @abstractmethod
    def n_actions(self) -> int:
        pass

    @property
    @abstractmethod
    def observation_shape(self) -> tuple:
        pass

    @abstractmethod
    def reset(self) -> State:
        pass

    @abstractmethod
    def step(self, action) -> Tuple[np.ndarray, float, bool, dict]:
        pass

    @abstractmethod
    def render(self) -> None:
        pass
