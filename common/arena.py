from __future__ import annotations
from abc import ABC,abstractmethod

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from common.network_wrappers import NNWrapper
from common.nnmcts import NNMCTS
from common.game import Game, State
import numpy as np
class Player(ABC):
    def __init__(self) -> None:
        pass
    @abstractmethod
    def choose_action(self,state:State)->int:
        pass

class Human(Player):
    def __init__(self) -> None:
        super().__init__()
    
    def choose_action(self, state: State) -> int:
        state.render()
        a = int(input('Choose Action \n'))
        return a

class NNMCTSPlayer(Player):
    def __init__(self,game:Game,nnetwrapper:NNWrapper,args) -> None:
        super().__init__()
        self.game = game
        self.nn_wrapper = nnetwrapper
        self.args = args
    
    def choose_action(self, state: State) -> int:
        mcts = NNMCTS(self.game,self.nn_wrapper,self.args)
        probs = mcts.probs(state,0.5)
        a = np.random.choice(len(probs),p=probs)
        return a

class NNPlayer(Player):
    def __init__(self,nn:NNWrapper) -> None:
        super().__init__()
        self.nn = nn
    
    def choose_action(self, state: State) -> int:
        probs,v = self.nn.predict(state.to_obs())
        legal_actions = state.get_legal_actions()
        probs = [probs[a] if a in legal_actions else 0 for a in range(len(probs))]
        probs_sum = float(sum(probs))
        norm_probs = [x/probs_sum for x in probs]
        action = np.random.choice(len(norm_probs),p=norm_probs)
        return action

class Arena():
    def __init__(self,player_1:Player,player_2:Player,game:Game,n_games=1,render=False) -> None:
        self.player_1 = player_1
        self.player_2 = player_2
        self.game = game
        self.n_games =n_games
        self.render = render
        self.winnings = np.zeros((2,),dtype=np.float32)
    
    def brawl(self)->float:
        starting_player = 0
        for i in range(self.n_games):
            s1,s2 = self._play_game(starting_player)
            self.winnings += (s1,s2)
            starting_player = 1-starting_player
        return self.winnings[0] / self.n_games
        
    def _play_game(self,starting_player)->tuple[float,float]:
        players = [self.player_1,self.player_2]
        scores = np.zeros((2,),dtype=np.float32)
        state = self.game.reset()
        done = False
        current_player = starting_player
        while True:
            player = players[current_player]
            a = player.choose_action(state)
            if a not in state.get_legal_actions():
                print(f'player {current_player+1} choose wrong action {a}\n')
                continue
            new_state : State
            new_state , reward,done,info = self.game.step(a)
            if done:
                assert new_state.is_game_over()
                result = new_state.game_result()
                scores[1-current_player] = result
                scores[current_player] = -result
                break
            state = new_state
            current_player = 1-current_player
        if scores[0] > scores[1]:
            return (1,0)
        elif scores[1] > scores[0]:
            return (0,1)
        else:
            return (0.5,0.5)
