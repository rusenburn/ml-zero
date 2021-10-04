from __future__ import annotations
from typing import List
from common.game import Game, State
import numpy as np

class TicTacToeState(State):
    def __init__(self, observation: np.ndarray) -> None:
        super().__init__()
        self._observation = observation

    @property
    def shape(self):
        return self._observation.shape

    @property
    def n_actions(self) -> int:
        return 9

    def get_legal_actions(self):
        legal_actions: List[int] = []
        for i in range(self.n_actions):
            if self._is_legal_action(i):
                legal_actions.append(i)
        return legal_actions

    def is_game_over(self) -> bool:
        player_0: int = 0
        player_1: int = 1
        return self._is_winning(player_0) or self._is_winning(player_1) or self._is_full()

    def game_result(self) -> int:
        player = self._observation[2][0][0]
        other = (player + 1) % 2
        if self._is_winning(player):
            return 1
        elif self._is_winning(other):
            return -1
        return 0

    def move(self, action):
        player = self._observation[2][0][0]
        next_player = (player + 1) % 2
        new_obs = self._observation.copy()
        new_obs[2] = next_player
        action_row = int(action // 3)
        action_col = int(action % 3)
        new_obs[player][action_row][action_col] = 1
        return TicTacToeState(new_obs)

    def render(self):
        player = self._observation[2][0][0]
        player_rep = ''
        if player == 0:
            player_rep = 'x'
        else:
            player_rep = 'o'
        result: List[str] = []
        result.append('****************************\n')
        result.append(f'*** Player {player_rep} has to move ***\n')
        result.append('****************************\n')
        result.append('\n')
        for row in range(3):
            for col in range(3):
                if self._observation[0][row][col] == 1:
                    result.append(' x ')
                elif self._observation[1][row][col] == 1:
                    result.append(' o ')
                else:
                    result.append(' . ')
                if col == 2:
                    result.append('\n')
        result.append('\n')
        print(''.join(result))

    def to_obs(self) -> np.ndarray:
        return self._observation.copy()

    def to_short(self) -> tuple:
        player = self._observation[2][0][0]
        space: np.ndarray = self._observation[0] - self._observation[1]
        return (player, *space.copy().flatten(),)

    def _is_winning(self, player):
        return self._is_horizontal_win(player) or \
            self._is_vertical_win(player) or \
            self._is_forward_win(player) or \
            self._is_backward_win(player)

    def _is_vertical_win(self, player) -> bool:
        vertical = (np.sum(self._observation[player, 0, :]) == 3 or
                    np.sum(self._observation[player, 1, :]) == 3 or
                    np.sum(self._observation[player, 2, :]) == 3)
        return vertical

    def _is_horizontal_win(self, player) -> bool:
        horizontal = (np.sum(self._observation[player, :, 0]) == 3 or
                      np.sum(self._observation[player, :, 1]) == 3 or
                      np.sum(self._observation[player, :, 2]) == 3)
        return horizontal

    def _is_forward_win(self, player) -> bool:
        forward = self._observation[player, 0, 0] == 1 == self._observation[player,
                                                                            1, 1] == self._observation[player, 2, 2]
        return forward

    def _is_backward_win(self, player) -> bool:
        backward = self._observation[player, 0, 2] == 1 == self._observation[player,
                                                                             1, 1] == self._observation[player, 2, 0]
        return backward

    def _is_legal_action(self, action) -> bool:
        player_0 = 0
        player_1 = 1
        action_row = int(action//3)
        action_col = int(action % 3)
        return self._observation[player_0][action_row][action_col] == 0 and self._observation[player_1][action_row][action_col] == 0

    def _is_full(self) -> bool:
        player_0: int = 0
        player_1: int = 1
        for row in range(3):
            for col in range(3):
                if self._observation[player_0][row][col] == 0 and self._observation[player_1][row][col] == 0:
                    return False
        return True


class TicTacToeGame(Game):
    def __init__(self,) -> None:
        super().__init__()
        self._initialize_state()

    @property
    def n_actions(self) -> int:
        return 9

    @property
    def observation_shape(self) -> tuple:
        return self._state.shape

    def reset(self):
        self._initialize_state()
        return self._state

    def _initialize_state(self):
        self._state = TicTacToeState(np.zeros((3, 3, 3), dtype=np.int))

    def step(self, action) -> tuple[TicTacToeState, float, bool, dict]:
        new_state = self._state.move(action)
        done = new_state.is_game_over()
        reward = -new_state.game_result()
        self._state = new_state
        return new_state, reward, done, {}

    def render(self):
        self._state.render()
