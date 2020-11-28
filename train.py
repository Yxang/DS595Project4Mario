import mario_wrapper
from mellowmax_agent import Agent_MM
from dqn_agent import Agent_DQN
from ddqn_agent import Agent_DDQN
from dqn_agent_y import Agent_DQN_y

env = mario_wrapper.create_env(reward_type = "sparse")
agent = Agent_MM(env)
agent.train()
env.close()
