from games.connect4.trainers import Connect4PPOTrainer, Connect4Trainer
from games.tictactoe.trainers import NewTrainer, PastSelfTrainer, TicTacToePPOTrainer, TicTacToePPOTrainer_, TicTacToeTrainer

def test_main():
    trainer = TicTacToeTrainer(20,100,25,0.55,2.5e-4,2.5e-4)
    nnet = trainer.train()
    nnet.save_check_point('tmp','tic_zero')


def train_connect4():
    trainer = Connect4Trainer(20,50,25,0.55,1e-3)
    nnet = trainer.train()
    nnet.save_check_point()

def train_tictactoe_ppo():
    trainer = TicTacToePPOTrainer(n_max_steps=1e5,n_epochs=4,batch_size=20,n_batches=4,lr=2.5e-4,testing_intervals=50)
    nnet = trainer.train()
    nnet.save_check_point('tmp','ppo')

def train_tictactoe_pop3d():
    trainer = NewTrainer(n_max_steps=1e5,n_epochs=4,batch_size=20,n_batches=4,lr=2.5e-4,testing_intervals=50)
    nnet = trainer.train()
    nnet.save_check_point('tmp','pop3d')

def train_connect4_ppo():
    trainer = Connect4PPOTrainer(n_max_steps=1e5,n_epochs=4,batch_size=20,n_batches=4,lr=2.5e-4,testing_intervals=50)
    nnet = trainer.train()
    nnet.save_check_point('tmp','connect4_PPO')

def train_past_self_ppo():
    trainer = PastSelfTrainer(n_max_steps=1e5,n_epochs=4,batch_size=20,n_batches=4,lr=2.5e-4,testing_intervals=50)
    nnet = trainer.train()
    nnet.save_check_point('tmp','tictactoe_past_self')
if __name__ == '__main__':
    # test_main()
    # train_connect4()
    # train_tic_gelu()
    # train_tictactoe_ppo()
    train_tictactoe_pop3d()
    # train_connect4_ppo()
    # train_past_self_ppo()
