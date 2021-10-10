from os import path
from games.connect4.network_wrappers import Connect4ResNetWrapper
from games.connect4.game import ConnectFourGame
from common.utils import dotdict
from games.connect4.trainers import Connect4PPOTrainer
from games.tictactoe import TicTacToeGame
from games.tictactoe.network_wrappers import ResNetWrapper
from common.arena import Arena, Human, NNMCTSPlayer, NNPlayer
from games.tictactoe.networks import FilteredSharedResNetwork, TicTacToeFilteredNetworkBase
from games.tictactoe.trainers import NewTrainer, PastSelfTrainer, TicTacToePPOTrainer
from trainers.ppo.network_wrappers import PPONNWrapper


def main():
    game = TicTacToeGame()
    args = dotdict({
        'cpuct': 1,
        'numMCTSSims': 25
    })

    nnet_1 = ResNetWrapper(game)
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

    t = Connect4PPOTrainer()
    
    nn_2 = t.wrapper
    nn_2.load_check_point('tmp','connect4_PPO')
    # p2 = NNPlayer(nn_2)
    p2 = NNMCTSPlayer(game,nn_2,args=args)
    arena = Arena(p1,p2,game,100)
    f = arena.brawl()
    print(f)


def test_ppo():
    args = dotdict({
        'cpuct': 1,
        'numMCTSSims': 100
    })
    game = TicTacToeGame()
    trainer = NewTrainer()
    nn_1 = trainer.wrapper
    nn_1.load_check_point('tmp','pop3d')
    p1 = NNPlayer(nn_1)
    # p1 = Human()
    trainer_2 = PastSelfTrainer()
    nn_2 = trainer_2.wrapper
    nn_2.load_check_point('tmp','tictactoe_past_self')
    p2 = NNPlayer(nn_2)
    # nnet_2 = ResNetWrapper(game)
    # nnet_2.load_check_point('tmp','tic_zero')
    # p2 = NNMCTSPlayer(game, nnet_2,args)
    # p1 = Human()
    arena = Arena(p1,p2,game,100)
    fraction= arena.brawl()
    print(f'{fraction:0.2f}')

if __name__ == '__main__':
    # main()
    # test_connect4()
    test_ppo()
