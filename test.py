import numpy as np
import mario_wrapper
from mellowmax_agent import Agent_MM

env = mario_wrapper.create_env()
agent = Agent_MM(env, test = True)
rewards = []

for i in range(30):
    state = env.reset()
    done = False
    episode_reward = 0.0

    #playing one game
    while(not done):
        env.env.render()
        action = agent.make_action(state, test = True)
        state, reward, done, info = env.step(action)
        episode_reward += reward

    rewards.append(episode_reward)
print('Run %d episodes'%(total_episodes))
print('Mean:', np.mean(rewards))
