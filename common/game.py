from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np


class State(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def get_legal_actions(self)->List[int]:
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

    def move(self, action: int) -> State:
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
    def step(self, action) -> Tuple[State, float, bool, dict]:
        pass

    @abstractmethod
    def render(self) -> None:
        pass


class GameVec(ABC):
    def __init__(self) -> None:
        super().__init__()
        pass

    @property
    @abstractmethod
    def n_actions(self) -> int:
        pass

    @property
    @abstractmethod
    def observation_shape(self) -> tuple:
        pass

    @abstractmethod
    def reset(self) -> List[State]:
        pass

    @abstractmethod
    def step(self, actions: List[int]) -> Tuple[List[State],List[float],List[bool],List[dict]]:
        pass

    @abstractmethod
    def render(self) -> None:
        pass


class SyncGameVec(GameVec):
    def __init__(self, environments: List[Game]) -> None:
        super().__init__()
        if not environments:
            raise Exception("Argument environments cannot be None or Empty")
        self._envs = [env for env in environments]
        self.n_enviroments = len(environments)

    @property
    def n_actions(self) -> int:
        return self._envs[0].n_actions

    @property
    def observation_shape(self) -> tuple:
        return self._envs[0].observation_shape

    @property
    def n_envs(self)->int:
        return len(self._envs)

    def reset(self) -> List[State]:
        result = [i.reset() for i in self._envs]
        return result

    def step(self, actions: List[int]) -> Tuple[List[State],List[float],List[bool],List[dict]]:
        if len(actions) != len(self._envs):
            raise Exception(
                f"actions length are supposed to be equal to the number of environments ({len(self._envs)}).")
        new_states, rewards, dones, infos = [], [], [], []
        for act, env in zip(actions, self._envs):
            new_state, reward, done, info = env.step(act)
            if done:
                new_state = env.reset()
            new_states.append(new_state)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)
        return new_states,rewards,dones,infos

    def render(self) -> None:
        for i, env in enumerate(self._envs):
            print(f"Env no {i}.\n")
            env.render()
