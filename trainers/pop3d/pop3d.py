from typing import List
from torch.distributions.categorical import Categorical
from torch.functional import Tensor
import numpy as np
from torch.nn.utils.clip_grad import clip_grad_norm_
from common.networks import NNBase
from common.utils import get_device
from trainers.ppo.ppo import PPO
import torch as T


class POP3D(PPO):
    def __init__(self, game_fns, nn: NNBase, n_max_steps=100000, n_epochs=4, batch_size=20, n_batches=4, lr=0.00025, gamma=0.99, gae_lambda=0.95, beta=5, testing_threshold=0.55, testing_intervals=50, min_lr=0.00001) -> None:
        super().__init__(game_fns, nn, n_max_steps=n_max_steps, n_epochs=n_epochs, batch_size=batch_size, n_batches=n_batches, lr=lr, gamma=gamma,
                         gae_lambda=gae_lambda, policy_clip=0.2, testing_threshold=testing_threshold, testing_intervals=testing_intervals, min_lr=min_lr)
        self.beta = beta
    def _train(self, examples: List,last_values:np.ndarray):
        self.wrapper.nn.train()
        for _ in range(self.n_epochs):
            observations_batches, action_batches, log_probs_batches, value_batches, reward_batches, done_batches = examples
            normalized_advatages,all_returns = self._calculate_advantages_improved(
                reward_batches, value_batches, done_batches,np.ndarray)
            batches = self._prepare_batches()

            states_arr, actions_arr, log_probs_arr, values_arr = self._reshape_batches(
                observations_batches, action_batches, log_probs_batches, value_batches)

            values = T.tensor(values_arr.copy(),device=get_device())
            states_tensor = T.tensor(states_arr,dtype=T.float32,device=get_device())
            with T.no_grad():
                all_probs,_ = self.wrapper.nn(states_tensor)
            
            fixed_probs:Tensor = self.fix_probs(states_tensor,all_probs)
            for batch in batches:
                observations = T.tensor(states_arr[batch], dtype=T.float32,device=get_device())
                old_logprobs = T.tensor(log_probs_arr[batch], dtype=T.float32,device=get_device())
                actions = T.tensor(actions_arr[batch],device=get_device())
                probs: Tensor
                critic_value: Tensor
                probs, critic_value = self.wrapper.nn(observations)
                dist: Categorical = Categorical(probs)

                entropy: Tensor = dist.entropy().mean()
                critic_value = critic_value.squeeze()
                
                new_logprobs: Tensor = dist.log_prob(actions)
                prob_ratio = (new_logprobs-old_logprobs).exp()
                ppd = (new_logprobs-old_logprobs) ** 2
                actor_loss = -(prob_ratio * normalized_advatages[batch]).mean() + self.beta * ppd.mean()

                a_loss_2 = (0.5* (fixed_probs[batch].detach() - probs)**2).mean()
                returns = all_returns[batch]
                old_values = values[batch]
                v_pred = critic_value 
                v_pred_clipped = old_values + T.clip(v_pred - old_values,-self.policy_clip,+self.policy_clip)
                vf_losses_1 = (v_pred - returns)**2
                vf_losses_2 = (v_pred_clipped - returns)**2
                vf_loss = (T.max(vf_losses_1,vf_losses_2)).mean()
                critic_loss = (returns - critic_value)**2
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + 0.5 * vf_loss - 0.01 * entropy + a_loss_2

                self.optimizer.zero_grad()

                total_loss.backward()

                if self.max_grad_norm:
                    clip_grad_norm_(self.wrapper.nn.parameters(),
                                    max_norm=self.max_grad_norm)
                self.optimizer.step()
    
    def fix_probs(self,state:Tensor,probs):
        a = state.numpy().copy()
        s_shape = state.shape
        valid_moves = np.logical_and(a[:, 0, :, :] == 0, a[:, 1, :, :] == 0).reshape(
            (s_shape[0], 9)).astype(np.float)
        valid_moves = T.tensor(valid_moves,dtype=T.float32,device=get_device())
        prob_1 = probs * valid_moves
        sum_v = T.sum(prob_1,dim=-1,keepdim=True)
        prob_1 = prob_1 / sum_v
        return prob_1
