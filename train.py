import mario_wrapper
from mellowmax_agent import Agent_MM
from dqn_agent import Agent_DQN
from ddqn_agent import Agent_DDQN
from dqn_agent_y import Agent_DQN_y

#PER Agents
from dqn_agent_PER import Agent_DQN_PER
from ddqn_agent_PER import Agent_DDQN_PER
#from mellowmax_agent_PER import Agent_MM_PER

env = mario_wrapper.create_env(reward_type = "sparse")
agent = Agent_DDQN_PER(env)
agent.train()
env.close()
