#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd


class DQN(nn.Module):

    def __init__(self, device, h=84, w=84, outputs=4):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(4, 64, kernel_size=8, stride=4, bias=False)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, bias=False)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, bias=False)
        self.fc1 = nn.Linear(128 * 7 * 7, 1024)
        self.fc2 = nn.Linear(1024, outputs)
        self.device = device
        self.to(device)

    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            m.bias.data.fill_(0.0)

        if type(m) == nn.Conv2d:
            torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            # m.bias.data.fill_(0.1)

    def forward(self, x):
        #print(x.shape)
        #x = x.permute(0, 3, 1, 2).contiguous()
        #x = x.to(self.device).float() / 255.
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    def save(self, path):
        torch.save(self.state_dict(), path)

