#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random
import numpy as np
from collections import deque, namedtuple
import os
import sys
import math
from itertools import count, islice
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import copy

from dqn_model_y import DQN
"""
you can import any package and define any extra function as you need
"""

torch.manual_seed(34)
np.random.seed(34)
random.seed(34)

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu:0')


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
        self.weights = deque(maxlen=capacity)

    def push(self, weight, *args):
        self.memory.append(Transition(*args))
        self.weights.append(weight)

    def sample(self, batch_size):
        indices = np.random.choice(np.arange(len(self.memory)), batch_size,
                                   p=np.abs(self.weights) / np.sum(np.abs(self.weights)))
        result = [self.memory[i] for i in indices]
        return result

    def update_weights(self, agent):
        bs = 2048
        for i in range(1, (len(self.memory) // bs) + 1):
            l = bs * (i - 1)
            r = min(bs * i, len(self.memory))
            batch = Transition(*zip(*islice(self.memory, l, r)))

            non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                    batch.next_state)), device=agent.device, dtype=torch.bool)
            non_final_next_states = torch.cat([torch.tensor(s).float().to(agent.device) for s in batch.next_state
                                                 if s is not None])
            reward_batch = torch.cat(batch.reward)

            next_state_values = torch.zeros(r - l).to(agent.device)
            with torch.no_grad():
                next_state_values[non_final_mask] = agent.net(non_final_next_states).detach().max(1)[0]

            expected_state_action_values = (next_state_values * agent.GAMMA) + reward_batch - next_state_values
            expected_state_action_values = expected_state_action_values.detach().view(-1).tolist()
            for j in range(l, r):
                self.weights[j] = expected_state_action_values[j - l]


    def __len__(self):
        return len(self.memory)

class Agent_DQN_y():
    def __init__(self, env, args=None):
        """
        Initialize everything you need here.
        For example:
            paramters for neural network
            initialize Q net and target Q net
            parameters for repaly buffer
            parameters for q-learning; decaying epsilon-greedy
            ...
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        self.env = env
        self.n_actions = env.action_space.n
        self.capacity = 100000#args['capacity']
        self.memory = ReplayMemory(self.capacity)
        self.position = 0


        self.BATCH_SIZE = 32#args['batch_size']
        self.GAMMA = 0.99#args['gamma']
        self.EPS_START = 0.99#args['eps_start']
        self.EPS_END = 0.05#args['eps_end']
        self.EPS_DECAY = 10000#args['eps_decay']
        self.TARGET_UPDATE = 50#args['target_update']
        self.WEIGHT_UPDATE = 10

        self.steps_done = 0
        self.history = []

        self.device = device
        self.net = DQN(self.device, outputs=self.n_actions)
        self.target_net = DQN(self.device, outputs=self.n_actions)

        self.optimizer = optim.Adam(self.net.parameters(), lr=6.25e-5)

        #f_name = 'best_ctd/x_32_92598.pth'
        #print(f'loading model {f_name}')
        #self.net.load_state_dict(torch.load(f_name))

        """if args.test_dqn:
            #you can load your model here
            print('loading trained model')
            ###########################
            # YOUR IMPLEMENTATION HERE #
            #f_name = 'Lcnn_20201104_50/x_32_100753.pth'
            f_name = 'Lcnn_20201105_50_10w/x_32_104325.pth'
            print(f'loading model {f_name}')
            self.net.load_state_dict(torch.load(f_name))"""
        self.target_net.load_state_dict(copy.deepcopy(self.net.state_dict()))
        self.target_net.eval()


    def init_game_setting(self):
        """
        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary.
        If no parameters need to be initialized, you can leave it as blank.
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        ###########################
        pass


    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent
        Input:
            observation: np.array
                stack 4 last preprocessed frames, shape: (84, 84, 4)
        Return:
            action: int
                the predicted action from trained model
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        if not test:
            sample = random.random()
            eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
                            math.exp(-1. * self.steps_done / self.EPS_DECAY)
            if sample > eps_threshold:
                with torch.no_grad():
                    state = torch.tensor(observation).float().to(self.device)
                    action = self.net(state).max(1)[1]
            else:
                action = torch.tensor([[random.randrange(self.n_actions)]]).long()
            action = action.detach().item()
        else:
            with torch.no_grad():
                state = torch.tensor(observation).float().to(self.device)
                action = self.net(state).max(1)[1].view(1, 1).detach().item()
        ###########################
        return action

    def push(self, weight, *args):
        """ You can add additional arguments as you need.
        Push new data to buffer and remove the old one if the buffer is full.

        Hints:
        -----
            you can consider deque(maxlen = 10000) list
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        self.memory.push(weight, *args)
        ###########################


    def replay_buffer(self, batch_size):
        """ You can add additional arguments as you need.
        Select batch from buffer.
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        return self.memory.sample(batch_size)
        ###########################


    def train(self, n_episodes=200000):
        """
        Implement your training algorithm here
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        best_reward = -1
        best_ep = -1
        for i_episode in range(n_episodes):
            # Initialize the environment and state
            observation = self.env.reset()
            state = observation
            reward_sum = 0.
            for t in count():
                # Select and perform an action
                action = self.make_action(state, test=False)
                observation, reward, done, _ = self.env.step(action)
                reward_sum += reward
                reward = torch.tensor([reward], device=self.device)

                if not done:
                    next_state = observation
                else:
                    next_state = None
                #print(observation)
                # Store the transition in memory
                if done:
                    weight = reward.detach().item()
                else:
                    with torch.no_grad():
                        weight = (self.net(torch.tensor(next_state).float().to(self.device))
                                  .detach().max(1)[0].item())
                self.push(weight, state, action, next_state, reward)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the target network)
                if t % 4 == 0:
                    self.optimize()
                if done:
                    self.history.append(reward_sum)
                    mean_reward = np.sum(self.history[-50:]) / 50
                    eps = self.EPS_END + \
                          (self.EPS_START - self.EPS_END) * math.exp(-1. * self.steps_done / self.EPS_DECAY)
                    print(f'EPISODE {i_episode}, REWARD {int(reward_sum):3d}, MEAN {mean_reward:6.3f}, STEPS {t}, '
                          f'BEST {best_reward} @ {best_ep}, '
                          f'EPSILON {eps}')
                    self.steps_done += 1

                    if mean_reward > best_reward and i_episode > 200:
                        print(f'NEW BEST MOVING REWARD: {mean_reward:6.3}')
                        torch.save(self.net.state_dict(), f'Lcnn_20201105_50_10w/x_{self.BATCH_SIZE}_{i_episode}.pth')
                        best_reward = mean_reward
                        best_ep = i_episode
                    break
            # Update the target network, copying all weights and biases in DQN
            if i_episode % self.TARGET_UPDATE == 0:
                self.target_net.load_state_dict(copy.deepcopy(self.net.state_dict()))
            if i_episode % self.WEIGHT_UPDATE == 0:
                self.memory.update_weights(self)
        torch.save(self.net.state_dict(), f'Lcnn_20201105_50_10w/x_{self.BATCH_SIZE}_final.pth')
        ###########################
    def optimize(self):
        if len(self.memory) < self.BATCH_SIZE:
            return
        transitions = self.replay_buffer(self.BATCH_SIZE)

        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.device, dtype=torch.bool)

        non_terminal_next_state = [torch.tensor(s).float().to(self.device) for s in batch.next_state
                                           if s is not None]
        if len(non_terminal_next_state) > 0:
            non_final_next_states = torch.cat(non_terminal_next_state)
        state_batch = torch.cat([torch.tensor(s).float().to(self.device) for s in batch.state])
        action_batch = torch.tensor(batch.action).reshape(-1, 1).to(self.device)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.BATCH_SIZE).to(self.device)

        if len(non_terminal_next_state) > 0:
            with torch.no_grad():
                next_state_values[non_final_mask] = self.target_net(non_final_next_states).detach().max(1)[0]

        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch
        expected_state_action_values = expected_state_action_values.unsqueeze(1)

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values.float(), expected_state_action_values.float())
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.net.parameters():
            param.grad.data.clamp_(-1., 1.)
        self.optimizer.step()

