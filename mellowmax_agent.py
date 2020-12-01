import random
import numpy as np
from collections import deque, namedtuple
import os
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import torch.optim as optim

from dqn_model import DQN

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state', 'done'))

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def mellowmax(values, omega = 1.0, axis = 1):
    n = values.shape[axis]
    return (torch.logsumexp(omega * values, axis=axis) - np.log(n)) / omega

class Agent_MM():
    def __init__(self, env, test = False):
        self.cuda = torch.device('cuda')
        print("Using device: " + torch.cuda.get_device_name(self.cuda), flush = True)

        self.env = env
        self.state_shape = env.observation_space.shape
        self.n_actions = env.action_space.n

        self.memory = deque(maxlen = 250000)
        self.batch_size = 32
        self.mem_threshold = 50000

        self.gamma = 0.99

        self.learning_rate = 1e-4

        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_period = 10000
        self.epsilon_decay = (self.epsilon - self.epsilon_min) / self.epsilon_period

        self.update_rate = 4
        self.omega = 100

        self.start_epoch = 1
        self.epochs = 1
        self.epoch = 20000

        self.model = DQN(self.state_shape, self.n_actions).to(self.cuda)
        print("DQN parameters: {}".format(count_parameters(self.model)))

        self.target = DQN(self.state_shape, self.n_actions).to(self.cuda)
        self.target.eval()
        self.target_update = 10000

        self.optimizer = optim.Adam(self.model.parameters(), lr = self.learning_rate)

        if test:
            self.model.load_state_dict(torch.load('model.pt'))

    def init_game_setting(self):
        pass


    def make_action(self, observation, test=False):
        epsilon = 0.01 if test else self.epsilon
        # turn action into tensor
        observation = torch.tensor(observation, device=self.cuda, dtype = torch.float)
        # turn off learning
        self.model.eval()
        # epsilon greedy policy
        if random.random() > epsilon:
            # no need to calculate gradient
            with torch.no_grad():
                # choose highest value action
                b = self.model(observation)
                b = b.cpu().data.numpy()
                action = np.random.choice(np.flatnonzero(np.isclose(b, b.max())))
        else:
            # random action
            action = random.choice(np.arange(self.n_actions))
        # turn learning back on
        self.model.train()
        return action

    def replay_buffer(self):
        # Return tuple of sars transitions
        states, actions, rewards, next_states, dones = zip(*random.sample(self.memory, self.batch_size))
        states = torch.tensor(np.vstack(states), device = self.cuda, dtype = torch.float)
        actions = torch.tensor(np.array(actions), device = self.cuda, dtype = torch.long)
        rewards = torch.tensor(np.array(rewards, dtype = np.float32), device = self.cuda, dtype = torch.float)
        next_states = torch.tensor(np.vstack(next_states), device = self.cuda, dtype = torch.float)
        dones = torch.tensor(np.array(dones, dtype = np.float32), device = self.cuda, dtype = torch.float)
        return states, actions, rewards, next_states, dones


    def experience_replay(self, n = 0):
        # clamp gradient
        clamp = False
        # Reset gradient (because it accumulates by default)
        self.optimizer.zero_grad()
        # sample experience memory
        states, actions, rewards, next_states, dones = self.replay_buffer()
        # get Q(s,a) for sample
        Q = self.model(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
        # Mellowmax
        Q_prime = mellowmax(self.model(next_states).detach(), self.omega, 1)
        # calculate y = r + gamma * max_a' Q(s',a') for non-terminal states
        Y = rewards + (self.gamma * Q_prime) * (1 - dones)
        # Huber loss of Q and Y
        loss = F.smooth_l1_loss(Q, Y)
        # Compute dloss/dx
        loss.backward()
        # Clamp gradient
        if clamp:
            for param in self.model.parameters():
                param.grad.data.clamp_(-1, 1)
        # Change the weights
        self.optimizer.step()

    def train(self):
        step = 0
        learn_step = 0
        print("Begin Training:", flush = True)
        learn_curve = []
        last30 = deque(maxlen = 30)
        for epoch in range(self.start_epoch, self.epochs + 1):
            durations = []
            rewards = []
            # progress bar
            epoch_bar = tqdm(range(self.epoch), total = self.epoch, ncols = 200)
            for episode in epoch_bar:
                # reset state
                state = self.env.reset()
                # decay epsilon
                if self.epsilon > self.epsilon_min:
                    self.epsilon -= self.epsilon_decay
                # run one episode
                done = False
                ep_duration = 0
                ep_reward = 0
                while not done:
                    step += 1
                    ep_duration += 1
                    # get epsilon-greedy action
                    action = self.make_action(state)
                    # do action
                    next_state, reward, done, info = self.env.step(action)
                    ep_reward += reward
                    # add transition to replay memory
                    self.memory.append(Transition(state, action, reward, next_state, done))
                    state = next_state
                    # learn from experience, if available
                    if step % self.update_rate == 0 and len(self.memory) > self.mem_threshold:
                        self.experience_replay(learn_step)
                        learn_step += 1
                    # update target network
                    if step % self.target_update == 1:
                        self.target.load_state_dict(self.model.state_dict())

                durations.append(ep_duration)
                rewards.append(ep_reward)
                last30.append(ep_reward)
                learn_curve.append(np.mean(last30))
                epoch_bar.set_description("epoch {}/{}, avg duration = {:.2f}, avg reward = {:.2f}, last30 = {:2f}".format(epoch, self.epochs, np.mean(durations), np.mean(rewards), learn_curve[-1]))
            # save model every epoch
            torch.save(self.model.state_dict(), 'model.pt')
            plt.clf()
            plt.plot(learn_curve)
            plt.title("MellowMax Epoch {}".format(epoch))
            plt.xlabel('Episodes')
            plt.ylabel('Moving Average Reward')
            plt.savefig("epoch{}.png".format(epoch))
            learn_curve = []
