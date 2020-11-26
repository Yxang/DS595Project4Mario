import torch
import torch.nn as nn

def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform_(m.weight)
        nn.init.constant_(m.bias, 0)

    
    elif type(m) == nn.LSTMCell:
    
        nn.init.constant_(m.bias_ih, 0)
        nn.init.constant_(m.bias_hh, 0)

class a3c(nn.Module):

    def __init__(self, ini_channel, action_space):
        super(a3c,self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(ini_channel,32,kernel_size=8,stride=4),
            nn.ReLU(),
            nn.Conv2d(32,64,kernel_size=4,stride=2),
            nn.ReLU(),
            nn.Conv2d(64,64,kernel_size=3,stride=1),
            nn.ReLU()
        )
        self.conv.apply(init_weights)

        self.lstmCell = nn.LSTMCell(64*7*7, 512)
        init_weights(self.lstmCell)
        self.critic_linear = nn.Linear(512, 1)
        init_weights(self.critic_linear)
        self.actor_linear = nn.Linear(512, action_space)
        init_weights(self.actor_linear)
    
    def forward(self,x,hx,cx):
        x = self.conv(x)
        hx,cx = self.lstmCell(x.view(x.size()[0],-1),(hx,cx))

        return self.actor_linear(hx), self.critic_linear(hx), hx,cx



