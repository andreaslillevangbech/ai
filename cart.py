#!/usr/bin/env python3
#import gym
from muzero.follower import Follower
import random

def randint(n):
  return random.randint(0, n-1)

env = Follower()
for i_episode in range(20):
  env.reset()
  for _ in range(100):
    env.render()
    env.step(randint(env.action_space.n))
    # env.step(env.action_space.sample() # take a random action
    if env.done:
      print("episode finished")
      break
env.close()
