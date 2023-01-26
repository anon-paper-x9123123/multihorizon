from collections import namedtuple
import random
from functools import partial
from types import SimpleNamespace
from typing import Tuple, Union

import numpy as np
import torch
import wandb
from torch import nn as nn
from torch.cuda.amp import GradScaler, autocast
import math

from common import networks
from common.replay_buffer import UniformReplayBuffer, PrioritizedReplayBuffer
from common.utils import prep_observation_for_qnet, compute_eval_gamma_interval, integrate_q_values

class Rainbow:
    buffer: Union[UniformReplayBuffer, PrioritizedReplayBuffer]

    def __init__(self, env, args: SimpleNamespace) -> None:
        self.env = env
        self.save_dir = args.save_dir
        self.use_amp = args.use_amp

        net = networks.get_model(args.network_arch, args.spectral_norm)
        linear_layer = partial(networks.FactorizedNoisyLinear, sigma_0=args.noisy_sigma0) if args.noisy_dqn else nn.Linear
        depth = args.frame_stack*(1 if args.grayscale else 3)   # greyscale or color
        self.q_policy = net(depth, env.action_space.n, linear_layer).cuda()
        self.q_target = net(depth, env.action_space.n, linear_layer).cuda()
        self.q_target.load_state_dict(self.q_policy.state_dict())

        #k = 0
        #for parameter in self.q_policy.parameters():
        #    k += parameter.numel()
        #print(f'Q-Network has {k} parameters.')

        self.double_dqn = args.double_dqn

        self.prioritized_er = args.prioritized_er
        self.prioritized_er_eps = args.prioritized_er_eps
        if self.prioritized_er:
            self.buffer = PrioritizedReplayBuffer(args.burnin, args.buffer_size, args.prioritized_er_alpha, [args.gamma], args.n_step, args.parallel_envs, use_amp=self.use_amp)
        else:
            self.buffer = UniformReplayBuffer(args.burnin, args.buffer_size, [args.gamma], args.n_step, args.parallel_envs, use_amp=self.use_amp)

        self.n_step_gamma = args.gamma ** args.n_step

        self.max_grad_norm = args.max_grad_norm
        self.opt = torch.optim.Adam(self.q_policy.parameters(), lr=args.lr, eps=args.adam_eps)
        self.scaler = GradScaler(enabled=self.use_amp)

        self.decay_lr = args.lr_decay_steps is not None
        if self.decay_lr: self.scheduler = torch.optim.lr_scheduler.StepLR(self.opt, (args.lr_decay_steps*args.train_count)//args.parallel_envs, gamma=args.lr_decay_factor)

        loss_fn_cls = nn.MSELoss if args.loss_fn == 'mse' else nn.SmoothL1Loss
        self.loss_fn = loss_fn_cls(reduction=('none' if self.prioritized_er else 'mean'))

    def sync_Q_target(self) -> None:
        self.q_target.load_state_dict(self.q_policy.state_dict())

    @torch.no_grad()
    def reset_noise(self, net) -> None:
        for m in net.modules():
            if isinstance(m, networks.FactorizedNoisyLinear):
                m.reset_noise()

    @torch.no_grad()
    def disable_noise(self, net) -> None:
        for m in net.modules():
            if isinstance(m, networks.FactorizedNoisyLinear):
                m.disable_noise()

    def act(self, states, eps: float):
        """ computes an epsilon-greedy step with respect to the current policy self.q_policy """
        with torch.no_grad():
            with autocast(enabled=self.use_amp):
                states = prep_observation_for_qnet(torch.from_numpy(np.stack(states)), self.use_amp)
                action_values = self.q_policy(states, advantages_only=True)
                actions = torch.argmax(action_values, dim=1)
            if eps > 0:
                for i in range(actions.shape[0]):
                    if random.random() < eps:
                        actions[i] = self.env.action_space.sample()
            return actions.cpu()

    @torch.no_grad()
    def td_target(self, reward: float, next_state, done: bool):
        self.reset_noise(self.q_target)
        if self.double_dqn:
            best_action = torch.argmax(self.q_policy(next_state, advantages_only=True), dim=1)
            next_Q = torch.gather(self.q_target(next_state), dim=1, index=best_action.unsqueeze(1)).squeeze()
            return reward + self.n_step_gamma * next_Q * (1 - done)
        else:
            max_q = torch.max(self.q_target(next_state), dim=1)[0]
            return reward + self.n_step_gamma * max_q * (1 - done)

    def train(self, batch_size, beta=None) -> Tuple[float, float, float]:
        if self.prioritized_er:
            indices, weights, (state, next_state, action, reward, done) = self.buffer.sample(batch_size, beta)
            weights = torch.from_numpy(weights).cuda()
        else:
            state, next_state, action, reward, done = self.buffer.sample(batch_size)

        self.opt.zero_grad()
        with autocast(enabled=self.use_amp):
            td_est = torch.gather(self.q_policy(state), dim=1, index=action.unsqueeze(1)).squeeze()
            td_tgt = self.td_target(reward, next_state, done)

            if self.prioritized_er:
                td_errors = td_est-td_tgt
                new_priorities = np.abs(td_errors.detach().cpu().numpy()) + self.prioritized_er_eps  # er_eps (1e-6) is the epsilon in PER
                self.buffer.update_priorities(indices, new_priorities)

                losses = self.loss_fn(td_tgt, td_est)
                loss = torch.mean(weights * losses)
            else:
                loss = self.loss_fn(td_tgt, td_est)

        self.scaler.scale(loss).backward()

        self.scaler.unscale_(self.opt)
        grad_norm = nn.utils.clip_grad_norm_(list(self.q_policy.parameters()), self.max_grad_norm)
        self.scaler.step(self.opt)
        self.scaler.update()

        if self.decay_lr:
            self.scheduler.step()
        # print(td_est.size()) # torch.Size([256])
        # print(type(loss), loss.size()) # <class 'torch.Tensor'> torch.Size([])
        return td_est.mean().item(), loss.item(), grad_norm.item()

    def save(self, steps, **kwargs):
        save_path = (self.save_dir + f"/checkpoint_{steps}.pt")
        torch.save({**kwargs, 'state_dict': self.q_policy.state_dict(), 'steps': steps}, save_path)

        try:
            artifact = wandb.Artifact('saved_model', type='model')
            artifact.add_file(save_path)
            wandb.run.log_artifact(artifact)
            print(f'Saved model checkpoint at {steps} steps.')
        except Exception as e:
            print('[bold red] Error while saving artifacts to wandb:', e)

class MH_Rainbow:
    buffer: Union[UniformReplayBuffer, PrioritizedReplayBuffer]

    def __init__(self, env, args: SimpleNamespace) -> None:
        self.env = env
        self.save_dir = args.save_dir
        self.use_amp = args.use_amp
         
        # for hyperbolic
        self.num_gammas = args.mh_num_gammas
        self.gamma_max = args.mh_gamma_max
        self.hyp_exp = args.mh_hyp_exp
        self.integral_estimate = args.mh_integral_estimate
        self.acting_policy = args.mh_acting_policy
        self.alternate_priorities = args.mh_alternate_priorities
        self.eval_gammas = compute_eval_gamma_interval(self.gamma_max, self.hyp_exp, self.num_gammas)
        
        self.gammas = [math.pow(gamma, self.hyp_exp) for gamma in self.eval_gammas] # gamma^k, these are the bellman gammas 
        hyperbolic_vars = namedtuple("hyperbolic_vars", ["gammas", "eval_gammas", "integral_estimate"])
        
        net = networks.get_model(args.network_arch, args.spectral_norm, hyperbolic_vars(self.gammas, self.eval_gammas, self.integral_estimate))
        linear_layer = partial(networks.FactorizedNoisyLinear, sigma_0=args.noisy_sigma0) if args.noisy_dqn else nn.Linear
        depth = args.frame_stack*(1 if args.grayscale else 3) # greyscale or color     
        self.q_policy = net(depth, env.action_space.n, linear_layer, args.mh_num_gammas).cuda()
        self.q_target = net(depth, env.action_space.n, linear_layer, args.mh_num_gammas).cuda()
        self.q_target.load_state_dict(self.q_policy.state_dict())

        #k = 0
        #for parameter in self.q_policy.parameters():
        #    k += parameter.numel()
        #print(f'Q-Network has {k} parameters.')

        self.double_dqn = args.double_dqn

        self.prioritized_er = args.prioritized_er
        self.prioritized_er_eps = args.prioritized_er_eps
        if self.prioritized_er:
            self.buffer = PrioritizedReplayBuffer(args.burnin, args.buffer_size, args.prioritized_er_alpha, self.gammas, args.n_step, args.parallel_envs, use_amp=self.use_amp)
        else:
            self.buffer = UniformReplayBuffer(args.burnin, args.buffer_size, self.gammas, args.n_step, args.parallel_envs, use_amp=self.use_amp)

        self.n_step = args.n_step
        self.max_grad_norm = args.max_grad_norm
        self.opt = torch.optim.Adam(self.q_policy.parameters(), lr=args.lr, eps=args.adam_eps)
        self.scaler = GradScaler(enabled=self.use_amp)

        self.decay_lr = args.lr_decay_steps is not None
        if self.decay_lr: self.scheduler = torch.optim.lr_scheduler.StepLR(self.opt, (args.lr_decay_steps*args.train_count)//args.parallel_envs, gamma=args.lr_decay_factor)

        loss_fn_cls = nn.MSELoss if args.loss_fn == 'mse' else nn.SmoothL1Loss
        self.loss_fn = loss_fn_cls(reduction=('none' if self.prioritized_er else 'mean'))
    

    def sync_Q_target(self) -> None:
        self.q_target.load_state_dict(self.q_policy.state_dict())

    @torch.no_grad()
    def reset_noise(self, net) -> None:
        for m in net.modules():
            if isinstance(m, networks.FactorizedNoisyLinear):
                m.reset_noise()

    @torch.no_grad()
    def disable_noise(self, net) -> None:
        for m in net.modules():
            if isinstance(m, networks.FactorizedNoisyLinear):
                m.disable_noise()

    def act(self, states, eps: float):
        """ computes an epsilon-greedy step with respect to the current policy self.q_policy """
        with torch.no_grad():
            with autocast(enabled=self.use_amp):
                states = prep_observation_for_qnet(torch.from_numpy(np.stack(states)), self.use_amp)
                if self.acting_policy == "hyperbolic":
                    action_values = self.q_policy(states, advantages_only=True).hyp_q_vals
                elif self.acting_policy == "largest_gamma":
                    action_values = self.q_policy(states, advantages_only=True).q_vals[-1]
                    
                actions = torch.argmax(action_values, dim=1)
            if eps > 0:
                for i in range(actions.shape[0]):
                    if random.random() < eps:
                        actions[i] = self.env.action_space.sample()
            return actions.cpu()

    @torch.no_grad()
    def td_target(self, n_step_rewards: float, next_state, done: bool):
        self.reset_noise(self.q_target)
        targets = []
        q_vals = self.q_policy(next_state, advantages_only=True).q_vals
        target_q_vals = self.q_target(next_state).q_vals
        if self.double_dqn:
            for q_val, target_q_val, gamma, gamma_idx in zip(q_vals, target_q_vals, self.gammas, range(len(self.gammas))):               
                best_action = torch.argmax(q_val, dim=1)        # get best action from current network
                next_Q = torch.gather(target_q_val, dim=1, index=best_action.unsqueeze(1)).squeeze()    # get Q(s') from target network for best action
                n_step_gamma = gamma ** self.n_step
                gamma_n_step_reward = n_step_rewards[:, gamma_idx] # get batch_size rewards for particular gamma i.e. [256,5] -> [256] for gamma_idx
                targets.append(gamma_n_step_reward + n_step_gamma * next_Q * (1 - done)) # n_step discounted reward + Y^n Q(s')
        else:   
            for q_val, target_q_val, gamma, gamma_idx in zip(q_vals, target_q_vals, self.gammas, range(len(self.gammas))):   
                max_q = torch.max(target_q_val, dim=1)[0]   # get best/max Q(s') from target network
                n_step_gamma = gamma ** self.n_step
                n_step_gamma_reward = n_step_rewards[:, gamma_idx]
                targets.append(n_step_gamma_reward + n_step_gamma * max_q * (1 - done)) # n_step discounted reward + Y^n Q(s') 
        
        return targets

    def train(self, batch_size, beta=None) -> Tuple[float, float, float]:
        if self.prioritized_er:
            indices, weights, (state, next_state, action, reward, done) = self.buffer.sample(batch_size, beta)
            weights = torch.from_numpy(weights).cuda()
        else:
            state, next_state, action, reward, done = self.buffer.sample(batch_size)

        self.opt.zero_grad()
        with autocast(enabled=self.use_amp):
            td_est = []
            q_vals = self.q_policy(state).q_vals
            for q_val in q_vals:
                td_est.append(torch.gather(q_val, dim=1, index=action.unsqueeze(1)).squeeze())  #td_est was [256], now it is [5,256] and not [256,5]
            td_tgt = self.td_target(reward, next_state, done) # list of td_targets
            
            total_loss = None
            total_priority = None

            if self.prioritized_er:
                # loop through outputs for each gamma head, calculate loss
                for est, tgt in zip(td_est, td_tgt):
                    td_error = est - tgt
                    new_priorities = np.abs(td_error.detach().cpu().numpy()) + self.prioritized_er_eps  # er_eps (1e-6) is the epsilon in PER
                    losses = self.loss_fn(tgt, est)
                    loss = torch.mean(weights * losses)
                    
                    # aggregate loss and priorities
                    if total_loss is None:
                        total_priority = new_priorities
                        total_loss = loss
                    else:
                        total_priority += new_priorities
                        total_loss += loss
                    
                # preserve scale of loss and priorities
                total_loss /= self.num_gammas

                if self.alternate_priorities:           # see Fedus (2019) Appendix E
                    total_priority = new_priorities     # set total_priority to the last calculated priority (which is TD-errors from largest gamma)
                else:
                    total_priority /= self.num_gammas   # prioritize by td-errors averaged evenly across all heads
                
                self.buffer.update_priorities(indices, total_priority)
            else:
                # loop through outputs for each gamma head, calculate loss
                for est, tgt in zip(td_est, td_tgt):
                    loss = self.loss_fn(tgt, est)
                    
                    # aggregate loss
                    if total_loss is None:
                        total_loss = loss
                    else:
                        total_loss += loss
                    
                # preserve scale of loss
                total_loss /= self.num_gammas                

        self.scaler.scale(total_loss).backward()

        self.scaler.unscale_(self.opt)
        grad_norm = nn.utils.clip_grad_norm_(list(self.q_policy.parameters()), self.max_grad_norm)
        self.scaler.step(self.opt)
        self.scaler.update()

        if self.decay_lr:
            self.scheduler.step()
        
        '''
        # report mean td_est (q-vals) across batch_size for last gamma_head only
        td_est_last = td_est[-1] # get td_est (q-vals) for only the largest gamma (just used for reporting in train.py)

        return td_est_last.mean().item(), total_loss.item(), grad_norm.item()

        '''
        # another option is to report mean of all q_vals (from each q_head)
        #print('axis0', len(td_est))        # 5 (num_gammas) , #print('axis1', len(td_est[0]))     # 256 (batch_size)
        td_est_all_heads = 0.
        for i in range(len(td_est)):
            td_est_all_heads += td_est[i].mean().item()     # calculate mean of each q_head, and add them up
        td_est_all_heads /= self.num_gammas                 # calculate mean td_est by dividing sum by num_gammas
        return td_est_all_heads, total_loss.item(), grad_norm.item()
        

    def save(self, steps, **kwargs):
        save_path = (self.save_dir + f"/checkpoint_{steps}.pt")
        torch.save({**kwargs, 'state_dict': self.q_policy.state_dict(), 'steps': steps}, save_path)

        try:
            artifact = wandb.Artifact('saved_model', type='model')
            artifact.add_file(save_path)
            wandb.run.log_artifact(artifact)
            print(f'Saved model checkpoint at {steps} steps.')
        except Exception as e:
            print('[bold red] Error while saving artifacts to wandb:', e)
            
    def _n_step_discount_vector(self, gamma):
        return gamma ** self.n_step
        
