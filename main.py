from games.connect4.trainers import Connect4Trainer
from games.tictactoe.trainers import GeluTrainerTest, TicTacToeTrainer

def test_main():
    trainer = TicTacToeTrainer(20,100,25,0.55,2.5e-4)
    nnet = trainer.train()
    nnet.save_check_point()


def train_tic_gelu():
    trainer = GeluTrainerTest(20,100,25,0.55,2.5e-4,2.5e-4)
    nnet = trainer.train()
    nnet.save_check_point('tmp','gelu_nn')

def train_connect4():
    trainer = Connect4Trainer(20,50,25,0.55,1e-3)
    nnet = trainer.train()
    nnet.save_check_point()
if __name__ == '__main__':
    # test_main()
    train_connect4()
    # train_tic_gelu()
