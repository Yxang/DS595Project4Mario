import mario_wrapper
from mellowmax_agent import Agent_MM
from dqn_agent import Agent_DQN
from ddqn_agent import Agent_DDQN

env = mario_wrapper.create_env()
agent = Agent_DDQN(env)
agent.train()
env.close()
