from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros._app.cli as cli
import gym_super_mario_bros
import cv2
import mario_wrapper
#from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT,COMPLEX_MOVEMENT
env = gym_super_mario_bros.make('SuperMarioBros-v0')
x = [['NOOP'],['right'],['A']]
# env = mario_wrapper.create_env()




done = True

for step in range(5000):
    if done:
        state = env.reset()
    state, reward, done, info = env.step(env.action_space.sample())

    print(state.shape)
    env.render()

env.close()










#cli.main()
