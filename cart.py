#!/usr/bin/env python3
import gym
from muzero.follower import Follower
import random


env = gym.make('LunarLander-v2')
for i_episode in range(20):
  observation = env.reset()
  for _ in range(10):
    #env.render()
    print(observation.shape)
    observation, reward, done, _ = env.step(env.action_space.sample()) # take a random action
    if done:
      print("episode finished")
      break
env.close()
