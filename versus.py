from os import path
from games.connect4.network_wrappers import Connect4ResNetWrapper
from games.connect4.game import ConnectFourGame
from common.utils import dotdict
from common.network_wrappers import TestResNetWrapper
from games.tictactoe import TicTacToeGame
from games.tictactoe.network_wrappers import ResNetWrapper
from common.arena import Arena, Human, NNMCTSPlayer, NNPlayer
from games.tictactoe.networks import FilteredSharedResNetwork, TicTacToeFilteredNetworkBase
from games.tictactoe.trainers import TicTacToePPOTrainer
from trainers.ppo.network_wrappers import PPONNWrapper


def main():
    game = TicTacToeGame()
    args = dotdict({
        'cpuct': 1,
        'numMCTSSims': 25
    })

    nnet_1 = TestResNetWrapper(game)
    nnet_1.load_check_point('tmp','gelu_nn')
    p1 = NNMCTSPlayer(game,nnet_1,args)
    nnet_2 = ResNetWrapper(game)
    nnet_2.load_check_point('tmp','res_nn_fixed')
    p2 = NNMCTSPlayer(game, nnet_2,args)
    # p2 = Human()

    arena = Arena(p1,p2,game,100)
    f = arena.brawl()
    print(f)

def test_connect4():
    args = dotdict({
        'cpuct': 1,
        'numMCTSSims': 100
    })
    game = ConnectFourGame()
    nn_1 = Connect4ResNetWrapper(game)
    nn_1.load_check_point('tmp','connect4_res_nn_20')
    p1 = NNMCTSPlayer(game,nn_1,args=args)

    
    p2 = Human()
    arena = Arena(p1,p2,game,5)
    f = arena.brawl()
    print(f)


def test_ppo():
    game = TicTacToeGame()
    trainer = TicTacToePPOTrainer()
    nn_1 = trainer.nn
    nn_1.load_check_point('tmp','nn')
    p1 = NNPlayer(nn_1)

    
    p2 = Human()
    arena = Arena(p1,p2,game,5)
    f= arena.brawl()
    print(f)

    
    pass

if __name__ == '__main__':
    # main()
    # test_connect4()
    test_ppo()
