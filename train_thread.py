#reference: https://arxiv.org/pdf/1506.02438.pdf

import os
import pickle
import random
import numpy as np
from collections import deque
from mario_wrapper import create_env
from torch.distributions import Categorical
import torch
import torch.nn.functional as F
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY
import sys

from a3c_model import a3c

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.manual_seed(595)
np.random.seed(595)
random.seed(595)
world = 1
stage = 1
actiontype = SIMPLE_MOVEMENT
tau = 1
gamma = 0.9
update = 200
max_episode = int(1e5)
n_step = 50
modelFile = 'a3c_checkpoint3.pth'
scoreFile = 'scores3.data'

def local_train(index,global_model,optimizer,save=False):
    
    torch.manual_seed(100 + index)
    done = True
    temp_step = 0
    
    env = create_env(world,stage)
    local_model = a3c(4,env.action_space.n)
    local_model.train()
  
    state = torch.from_numpy(env.reset())
   
    
    for i_episode in range(max_episode):




        if save:
            print(i_episode)
        local_model.load_state_dict(global_model.state_dict())
        if done:
            hx = torch.zeros((1,512),dtype=torch.float)
            cx = torch.zeros((1,512),dtype=torch.float)
        
        else:
            hx = hx.detach()
            cx = cx.detach()

        log_policies = []
        v = []
        r = []
        entropies = []

        for _ in range(n_step):
            temp_step+=1

            pi,value,hx,cx = local_model(state,hx,cx)
            policy = F.softmax(pi,dim=1)
          
            log_policy = F.log_softmax(pi,dim=1)
        
            entropy = -(policy*log_policy).sum(1)

            m = Categorical(policy)
            action = m.sample().item()    #stochastic policy

            state, reward, done, _ = env.step(action)
            state = torch.from_numpy(state)
            
            log_policies.append(log_policy[0,action])
            v.append(value)
            r.append(reward)
            entropies.append(entropy)

            if done:
                temp_step=0
                state = torch.from_numpy(env.reset())
                break

        actorL = 0
        criticL = 0
       

        R = torch.zeros((1,1),dtype=float)
        next_value = R
        if not done:
            _,R,_,_ =  local_model(state,hx,cx) # if the episode is end, we can calculate the R, if not, we use V to estimate the future R

        gae = torch.zeros((1,1),dtype=float) # we use generalized advantage estimator (gae) to estimate gradient
        for log_policy, value, reward, entropy in list(zip(log_policies,v,r,entropies))[::-1]:
            gae = gae*gamma*tau
            gae = gae + reward + gamma*next_value.detach() - value.detach()
            actorL = actorL - gae*log_policy - 0.01*entropy
            next_value = value

            R = gamma*R + reward
            criticL = criticL + 0.5*((R-value).pow(2))

        totalL= actorL + criticL
        optimizer.zero_grad()
        
        totalL.backward()

        for local_p, global_p in zip(local_model.parameters(),global_model.parameters()):
            if global_p.grad is not None:
                break

            
            global_p._grad = local_p.grad


        optimizer.step()
        if save:
           
            if i_episode%update== 0:
                torch.save(global_model.state_dict(), modelFile)

      

        

            
def local_test(index,global_model):
    scores = []

    if os.path.exists(scoreFile):
        print('score exist')
        with open(scoreFile, 'rb') as sc1:
            scores = pickle.load(sc1)



    torch.manual_seed(123 + index)
    done = True
    temp_step = 0
  
    env = create_env(world,stage)
    local_model = a3c(4,env.action_space.n)
    local_model.eval()
  
    n_step = 50
    state = torch.from_numpy(env.reset())
    
    R = 0
    rlist = deque(maxlen=30)
    while True:
        
        local_model.load_state_dict(global_model.state_dict())
        if done:
            hx = torch.zeros((1,512),dtype=torch.float)
            cx = torch.zeros((1,512),dtype=torch.float)
        
        else:
            hx = hx.detach()
            cx = cx.detach()

        R = 0

        for _ in range(n_step):
            temp_step+=1

            pi,_,hx,cx = local_model(state,hx,cx)
            policy = F.softmax(pi)


          
            action =  torch.argmax(policy).item()

            state, reward, done, _ = env.step(action)
            state = torch.from_numpy(state)
            
            R = R+reward
            env.render()
            if done:
                temp_step=0
                state = torch.from_numpy(env.reset())
                rlist.append(R)
                scores.append(np.mean(rlist))
                print("Mean score: {:.4f}  present score: {:.4f}".format(np.mean(rlist),R))
                with open(scoreFile, 'wb') as sc:
                        
                        pickle.dump(scores, sc)
              
                R = 0
                break

      
     