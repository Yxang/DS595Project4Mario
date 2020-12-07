import numpy as np
import mario_wrapper
from mellowmax_agent import Agent_MM
from dqn_agent_PER import Agent_DQN_PER
import time
from tqdm.auto import tqdm

env = mario_wrapper.create_env()
agent = Agent_DQN_PER(env, test = True)
rewards = []
flag = []

for i in tqdm(range(100)):
    state = env.reset()
    done = False
    episode_reward = 0.0

    #playing one game
    while(not done):
        #env.env.render()
        action = agent.make_action(state, test = True)
        state, reward, done, info = env.step(action)
        episode_reward += reward
        #time.sleep(1/80)
    flag.append(info['flag_get'])
    rewards.append(episode_reward)

print('Mean:', np.mean(rewards))
print("Flag rate", np.mean(flag))
