import collections
import random
from math import sqrt

import numpy as np
import torch
from gym.wrappers import LazyFrames

from common.utils import prep_observation_for_qnet


class UniformReplayBuffer:
    def __init__(self, burnin, capacity, gammas, n_step, parallel_envs, use_amp):
        self.capacity = capacity
        self.burnin = burnin
        self.buffer = []
        self.nextwrite = 0
        self.use_amp = use_amp

        self.gammas = gammas
        self.n_step = n_step
        self.n_step_buffers = [collections.deque(maxlen=self.n_step + 1) for j in range(parallel_envs)]

    def put(self, *transition, j):
        self.n_step_buffers[j].append(transition)
        if len(self.n_step_buffers[j]) == self.n_step + 1 and not self.n_step_buffers[j][0][3]:  # n-step transition can't start with terminal state
            state = self.n_step_buffers[j][0][0]
            action = self.n_step_buffers[j][0][1]
            next_state = self.n_step_buffers[j][self.n_step][0]
            done = self.n_step_buffers[j][self.n_step][3]
            num_gammas = len(self.gammas)
            reward = np.ones(num_gammas) * self.n_step_buffers[j][0][2]     # list of first reward
            # calculate n_step discounted rewards for each gamma            
            for i, gamma in zip(range(num_gammas), self.gammas):
                for k in range(1, self.n_step):
                    reward[i] += self.n_step_buffers[j][k][2] * gamma ** k  # build each-gamma-discounted n_step rewards
                    if self.n_step_buffers[j][k][3]:
                        done = True
                        break

            action = torch.LongTensor([action]).cuda()
            reward = torch.FloatTensor(reward).cuda()   # rewards is now already a list, else it throws a warning of slow execution
            done = torch.FloatTensor([done]).cuda()

            if len(self.buffer) < self.capacity:
                self.buffer.append((state, next_state, action, reward, done))
            else:
                self.buffer[self.nextwrite % self.capacity] = (state, next_state, action, reward, done)
                self.nextwrite += 1

    def sample(self, batch_size, beta=None):
        """ Sample a minibatch from the ER buffer (also converts the FrameStacked LazyFrames to contiguous tensors) """
        batch = random.sample(self.buffer, batch_size)
        state, next_state, action, reward, done = zip(*batch)
        state = list(map(lambda x: torch.from_numpy(x.__array__()), state))
        next_state = list(map(lambda x: torch.from_numpy(x.__array__()), next_state))

        state, next_state, action, reward, done = map(torch.stack, [state, next_state, action, reward, done])
        return prep_observation_for_qnet(state, self.use_amp), prep_observation_for_qnet(next_state, self.use_amp), \
               action.squeeze(), reward.squeeze(), done.squeeze()

    @property
    def burnedin(self):
        return len(self) >= self.burnin

    def __len__(self):
        return len(self.buffer)


class PrioritizedReplayBuffer:
    """ based on https://nn.labml.ai/rl/dqn, supports n-step bootstrapping and parallel environments,
    removed alpha hyperparameter like google/dopamine (https://github.com/google/dopamine/blob/a2753dae222c75ae991758d4110a84bc01c3215f/dopamine/agents/rainbow/rainbow_agent.py#L26)
    """
    def __init__(self, burnin, capacity, alpha, gammas, n_step, parallel_envs, use_amp):
        self.burnin = burnin
        self.capacity = capacity    # must be a power of two
        self.alpha = alpha          # determines how much prioritization is used for each transition       
        self.gammas = gammas        # list of gammas of len(num_gammas)
        self.n_step = n_step
        self.n_step_buffers = [collections.deque(maxlen=self.n_step + 1) for j in range(parallel_envs)]

        self.use_amp = use_amp

        self.priority_sum = [0 for _ in range(2 * self.capacity)]
        self.priority_min = [float('inf') for _ in range(2 * self.capacity)]

        self.max_priority = 1.0  # initial priority of new transition is set to maximum, so that it gets sampled

        self.data = [None for _ in range(self.capacity)]  # cyclical buffer for transitions
        self.next_idx = 0  # next write location
        self.size = 0  # number of buffer elements

    @staticmethod
    def prepare_transition(state, next_state, action: int, reward: float, done: bool):
        action = torch.LongTensor([action]).cuda()
        reward = torch.FloatTensor(reward).cuda()   # to avoid warning: rewards is now already a list i.e. [bs] -> [bs,num_gammas]
        done = torch.FloatTensor([done]).cuda()

        return state, next_state, action, reward, done

    def put(self, *transition, j):
        self.n_step_buffers[j].append(transition)   # self.n_step_buffers[j=parallel_env][k=n_step][l=state=0, action=1, reward=2, done=3]
        if len(self.n_step_buffers[j]) == self.n_step + 1 and not self.n_step_buffers[j][0][3]:  # n-step transition can't start with terminal state
            state = self.n_step_buffers[j][0][0]                    # starting state of n_step buffer is at k=0
            action = self.n_step_buffers[j][0][1]                   # starting action of n_step buffer is at k=0
            next_state = self.n_step_buffers[j][self.n_step][0]     # next state is now at k=n_step so Y^0 r0 + Y^1 r1 + Y^2 r2 + next state (for n_step=3)
            done = self.n_step_buffers[j][self.n_step][3]           # done is now read at n=n_step
            num_gammas = len(self.gammas)
            reward = np.ones(num_gammas) * self.n_step_buffers[j][0][2]  # get a list of first rewards [r_t0_y0, r_t0_y1, ..., r_t0_numgammas]
            # calculate n_step discounted rewards for each gamma
            for i, gamma in zip(range(num_gammas), self.gammas):
                for k in range(1, self.n_step):
                    reward[i] += self.n_step_buffers[j][k][2] * gamma ** k  # build each-gamma-discounted n_step rewards
                    if self.n_step_buffers[j][k][3]:
                        done = True
                        break
            
            assert isinstance(state, LazyFrames)
            assert isinstance(next_state, LazyFrames)

            idx = self.next_idx
            self.data[idx] = self.prepare_transition(state, next_state, action, reward, done)
            self.next_idx = (idx + 1) % self.capacity
            self.size = min(self.capacity, self.size + 1)
            priority_alpha = self.max_priority ** self.alpha    # bugfix: previously, each transition kept getting max_priority=1 [ sqrt(1)=1 ]
            self._set_priority_min(idx, priority_alpha)
            self._set_priority_sum(idx, priority_alpha)

    def _set_priority_min(self, idx, priority_alpha):
        idx += self.capacity
        self.priority_min[idx] = priority_alpha
        while idx >= 2:
            idx //= 2
            self.priority_min[idx] = min(self.priority_min[2 * idx], self.priority_min[2 * idx + 1])

    def _set_priority_sum(self, idx, priority):
        idx += self.capacity
        self.priority_sum[idx] = priority
        while idx >= 2:
            idx //= 2
            self.priority_sum[idx] = self.priority_sum[2 * idx] + self.priority_sum[2 * idx + 1]

    def _sum(self):
        return self.priority_sum[1]

    def _min(self):
        return self.priority_min[1]

    def find_prefix_sum_idx(self, prefix_sum):
        """ find the largest i such that the sum of the leaves from 1 to i is <= prefix sum"""

        idx = 1
        while idx < self.capacity:
            if self.priority_sum[idx * 2] > prefix_sum:
                idx = 2 * idx
            else:
                prefix_sum -= self.priority_sum[idx * 2]
                idx = 2 * idx + 1
        return idx - self.capacity

    def sample(self, batch_size: int, beta: float) -> tuple:
        weights = np.zeros(shape=batch_size, dtype=np.float32)
        indices = np.zeros(shape=batch_size, dtype=np.int32)

        for i in range(batch_size):
            p = random.random() * self._sum()
            idx = self.find_prefix_sum_idx(p)
            indices[i] = idx

        prob_min = self._min() / self._sum()
        max_weight = (prob_min * self.size) ** (-beta)

        for i in range(batch_size):
            idx = indices[i]
            prob = self.priority_sum[idx + self.capacity] / self._sum()
            weight = (prob * self.size) ** (-beta)
            weights[i] = weight / max_weight

        samples = []
        for i in indices:
            samples.append(self.data[i])

        return indices, weights, self.prepare_samples(samples)

    def prepare_samples(self, batch):
        state, next_state, action, reward, done = zip(*batch)
        state = list(map(lambda x: torch.from_numpy(x.__array__()), state))
        next_state = list(map(lambda x: torch.from_numpy(x.__array__()), next_state))

        state, next_state, action, reward, done = map(torch.stack, [state, next_state, action, reward, done])
        return prep_observation_for_qnet(state, self.use_amp), prep_observation_for_qnet(next_state, self.use_amp), \
               action.squeeze(), reward.squeeze(), done.squeeze()

    def update_priorities(self, indexes, priorities):
        for idx, priority in zip(indexes, priorities):
            self.max_priority = max(self.max_priority, priority)
            priority_alpha = priority ** self.alpha
            self._set_priority_min(idx, priority_alpha)
            self._set_priority_sum(idx, priority_alpha)

    @property
    def is_full(self):
        return self.capacity == self.size

    @property
    def burnedin(self):
        return len(self) >= self.burnin

    def __len__(self):
        return self.size

