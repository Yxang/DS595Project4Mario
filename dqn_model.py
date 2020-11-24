import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DQN(nn.Module):

    def __init__(self, state_shape, num_actions):
        super(DQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size = 8, stride = 4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size = 4, stride = 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size = 3, stride = 1),
            nn.ReLU())

        self.lin = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions))

    def forward(self, x):
        conv = self.conv(x).view(x.size(0), -1)
        lin = self.lin(conv)
        return lin
