import math
from common.network_wrappers import NNWrapper
from common.game import Game, State
import numpy as np

class NNMCTS():
    def __init__(self,game:Game,nnet:NNWrapper,args) -> None:
        self.game:Game = game
        self.nnet = nnet
        self.args = args
        self.visited = set()
        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}  # stores # times edge s,a was visited
        self.Ns = {}  # stores # times board s was visited
        self.Ps = {}  # stores initial policy (returned by neural net)

        self.Es = {}  # stores game.getGameEnded ended for board s
        self.Vs = {}  # stores game.getValidMoves for board s
    
    def search(self,state:State):
        if state.is_game_over():
            return -state.game_result()
        s = state.to_short()
        # s = (*state.to_obs().flatten(),)
        if s not in self.visited:
            self.visited.add(s)
            # self.visited.append(s)
            obs = state.to_obs()
            self.Ps[s], v = self.nnet.predict(obs)
            return -v
        
        max_u , best_a = -float("inf"), -1
        for a in state.get_legal_actions():
            if (*s,a) not in self.Nsa :
                self.Nsa[(*s,a)] = 0
            if s not in self.Ns:
                self.Ns[s] = 0
            if (*s,a) in self.Qsa:
                u = self.Qsa[(*s,a)] + self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (1 + self.Nsa[(*s,a)])
            else:
                u =  self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (1 + self.Nsa[(*s,a)])
            if u > max_u:
                max_u = u
                best_a = a
        
        a = best_a
        sp = state.move(a)
        v = self.search(sp)
        if (*s,a) in self.Qsa:
            self.Qsa[(*s,a)] = (self.Nsa[(*s,a)] * self.Qsa[(*s,a)] + v) / (self.Nsa[(*s,a)] + 1)
            self.Nsa[(*s,a)] +=1
        else:
            self.Qsa[(*s,a)] = v
            self.Nsa[(*s,a)] = 1
        b : bool = s not in self.Ns
        if b:
            b=False
        self.Ns[s] +=1
        return -v


    def probs(self,state:State,temp=1):
        assert not state.is_game_over()
        for i in range(self.args.numMCTSSims):
            self.search(state)
        # s = (*state.to_obs().flatten(),)
        s = state.to_short()
        # qs = [self.Qsa[(*s,a)] if (*s,a) in self.Qsa else 0 for a in range(self.game.n_actions)]
        # ps = self.Ps[s]
        counts = [self.Nsa[(*s,a)] if (*s,a)in self.Nsa else 0 for a in range(self.game.n_actions)]
        if temp == 0:
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            bestA = np.random.choice(bestAs)
            probs = [0]*len(counts)
            probs[bestA] = 1
            return probs
        counts = [x**(1./temp) for x in counts]
        counts_sum = float(sum(counts))
        probs = [x/counts_sum for x in counts]
        return probs