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
import pickle

from dqn_model import DQN
from PER import PriorityMemory

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state', 'done'))

save_prefix = "sparse"

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class Agent_DQN_PER():
    def __init__(self, env, test = False):
        self.cuda = torch.device('cuda')
        print("Using device: " + torch.cuda.get_device_name(self.cuda), flush = True)

        self.env = env
        self.state_shape = env.observation_space.shape
        self.n_actions = env.action_space.n

        #self.memory = deque(maxlen = 250000)
        self.memory = PriorityMemory(100000)
        self.batch_size = 32
        self.mem_threshold = 50000

        self.gamma = 0.99

        self.learning_rate = 1e-4

        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_period = 10000
        self.epsilon_decay = (self.epsilon - self.epsilon_min) / self.epsilon_period

        self.update_rate = 4

        self.start_epoch = 1
        self.epochs = 10
        self.epoch = 10000

        self.model = DQN(self.state_shape, self.n_actions).to(self.cuda)
        print("DQN parameters: {}".format(count_parameters(self.model)))

        self.target = DQN(self.state_shape, self.n_actions).to(self.cuda)
        self.target.eval()
        self.target_update = 10000

        self.optimizer = optim.Adam(self.model.parameters(), lr = self.learning_rate)
        if test:
            self.model.load_state_dict(torch.load('wrapper_PER_DQN/DQN_model_ep4.pt'))

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
        indices, data = map(list, zip(*self.memory.sample(self.batch_size)))    #Unzip indices and data
        states, actions, rewards, next_states, dones = zip(*data)               #Unzip data of transitions
        #states, actions, rewards, next_states, dones = zip(*random.sample(self.memory, self.batch_size))
        states = torch.tensor(np.vstack(states), device = self.cuda, dtype = torch.float)
        actions = torch.tensor(np.array(actions), device = self.cuda, dtype = torch.long)
        rewards = torch.tensor(np.array(rewards, dtype = np.float32), device = self.cuda, dtype = torch.float)
        next_states = torch.tensor(np.vstack(next_states), device = self.cuda, dtype = torch.float)
        dones = torch.tensor(np.array(dones, dtype = np.float32), device = self.cuda, dtype = torch.float)
        return states, actions, rewards, next_states, dones, indices


    def experience_replay(self, n = 0):
        # clamp gradient
        clamp = False
        # Reset gradient (because it accumulates by default)
        self.optimizer.zero_grad()
        # sample experience memory
        states, actions, rewards, next_states, dones, indices = self.replay_buffer()
        # get Q(s,a) for sample
        Q = self.model(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)

        # get max_a' Q(s',a')
        Q_prime = self.target(next_states).detach().max(1)[0]

        # calculate y = r + gamma * max_a' Q(s',a') for non-terminal states
        Y = rewards + (self.gamma * Q_prime) * (1 - dones)

        #Find errors for PER
        errors = torch.abs(Y - Q).data.cpu().numpy()
        for i in range(self.batch_size):
            self.memory.update(indices[i], errors[i])

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
            flag = []
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
                    #self.memory.append(Transition(state, action, reward, next_state, done))
                    self.memory.add(abs(reward), Transition(state, action, reward, next_state, done)) #Will overwrite once full
                    state = next_state
                    # learn from experience, if available
                    if step % self.update_rate == 0 and self.memory.writes > self.mem_threshold:
                        self.experience_replay(learn_step)
                        learn_step += 1
                    # update target network
                    if step % self.target_update == 1:
                        self.target.load_state_dict(self.model.state_dict())

                durations.append(ep_duration)
                rewards.append(ep_reward)
                last30.append(ep_reward)
                learn_curve.append(np.mean(last30))
                flag.append(info['flag_get'])
                epoch_bar.set_description("epoch {}/{}, avg duration = {:.2f}, avg reward = {:.2f}, last30 = {:2f}".format(epoch, self.epochs, np.mean(durations), np.mean(rewards), learn_curve[-1]))
            # save model every epoch
            plt.clf()
            plt.plot(learn_curve)
            plt.title(f"PER DQN Epoch {epoch} with {save_prefix} Reward")
            plt.xlabel('Episodes')
            plt.ylabel('Moving Average Reward')
            if not os.path.exists(f"{save_prefix}_PER_DQN"):
                os.mkdir(f"{save_prefix}_PER_DQN")
            torch.save(self.model.state_dict(), f'{save_prefix}_PER_DQN/DQN_model_ep{epoch}.pt')
            pickle.dump(rewards, open(f"{save_prefix}_PER_DQN/DQN_reward_ep{epoch}.pkl", 'wb'))
            pickle.dump(flag, open(f"{save_prefix}_PER_DQN/flag_ep{epoch}.pkl", 'wb'))
            plt.savefig(f"{save_prefix}_PER_DQN/epoch{epoch}.png")
            learn_curve = []
