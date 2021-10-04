from games.connect4.trainers import Connect4Trainer
from games.tictactoe.trainers import GeluTrainerTest, TicTacToePPOTrainer, TicTacToeTrainer

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

def train_tictactoe_ppo():
    trainer = TicTacToePPOTrainer(n_max_steps=1e5,n_epochs=4,batch_size=20,n_batches=4,lr=2.5e-4)
    nnet = trainer.train()
    nnet.save_check_point('tmp','ppo')
if __name__ == '__main__':
    # test_main()
    # train_connect4()
    # train_tic_gelu()
    train_tictactoe_ppo()
