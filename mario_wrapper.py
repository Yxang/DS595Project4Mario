

#reference: https://github.com/uvipen/Super-mario-bros-A3C-pytorch/blob/master/src/env.py

import gym_super_mario_bros
from gym.spaces import Box
from gym import Wrapper
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY
import cv2
import numpy as np



# transfer the frame into (1,84,84)
def transfer_frame(state):

    state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
    state = cv2.resize(state,(84,84))[None,:,:]/255
    return state


#The original env provided time panelty and distance reward already, but you can make change at here

class set_reward(Wrapper):
    def _init_(self,env=None):
        super(set_reward,self).__init__(env)
        self.observation_space = Box(low=0, high=255, shape=(1, 84, 84))

        self.curr_score = 0


    def step(self,action):
        state, reward, done, info = self.env.step(action)
        state = transfer_frame(state)

        reward += (info["score"] - self.curr_score) / 40
        self.curr_score = info["score"]


        reward += max(info['x_pos'] - self.x_pos,0)
        self.x_pos = info['x_pos']


        if done:
            if info["flag_get"]:
                reward += 50
            else:
                reward -= 50
        return state, reward / 5., done, info

    def reset(self):
        self.curr_score = 0
        self.x_pos = 0
        return transfer_frame(self.env.reset())

#Like breakout, merge 4 (1,84,84) frames into 1 (4,84,84), clip the reward between -15 and 15

class CustomState(Wrapper):
    def __init__(self, env, skip=4):
        super(CustomState, self).__init__(env)
        self.observation_space = Box(low=0, high=255, shape=(4, 84, 84))
        self.skip = skip

    def step(self, action):
        total_reward = 0
        states = []
        state, reward, done, info = self.env.step(action)
        for _ in range(self.skip):
            if not done:
                state, reward, done, info = self.env.step(action)
                total_reward += reward
                states.append(state)
            else:
                states.append(state)
        states = np.concatenate(states, 0)[None, :, :, :]
        return states.astype(np.float32), np.clip(total_reward,-15,15), done, info

    def reset(self):
        state = self.env.reset()

        states = np.concatenate([state for _ in range(self.skip)], 0)[None, :, :, :]
        return states.astype(np.float32)





#For Hierarchical training, separate the game into different segments - world and stage.
#I use game v2, which is down-scaled. v0 is original.

def create_env(world=1,stage=1,action_type=SIMPLE_MOVEMENT):

    env = gym_super_mario_bros.make('SuperMarioBros-{}-{}-v2'.format(world,stage))
    env = JoypadSpace(env, action_type)
    env = set_reward(env)
    env = CustomState(env)
    #print(env.skip)
    return env
