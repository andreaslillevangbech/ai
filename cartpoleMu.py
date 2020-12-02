import tensorflow as tf
import numpy as np
import gym
from tqdm import tqdm, trange
import os,sys

# Make Follower work! Will give interview to anyone who does.
from muzero.follower import Follower
#env = Follower()
env = gym.make("CartPole-v0")
#env = gym.make("MountainCar-v0")
#env = gym.make("LunarLander-v2")
#env = gym.make("Acrobot-v1")

from muzero.model import MuModel
m = MuModel(env.observation_space.shape, env.action_space.n, s_dim=128, K=3, lr=0.001)
print(env.observation_space.shape, env.action_space.n)

from muzero.game import Game, ReplayBuffer
from muzero.mcts import naive_search, mcts_search
replay_buffer = ReplayBuffer(50, 128, m.K)
rews = []

import random
def play_game(env, m):
    game = Game(env, discount=0.997)
    while not game.terminal():
        game.env.render()
        cc = random.random()
        if cc < 0.05:
            policy = [1/m.a_dim]*m.a_dim
        else:
            policy = naive_search(m, game.observation, T=1)
        game.act_with_policy(policy)
    return game

from muzero.model import reformat_batch
import collections

for j in range(30):
    game = play_game(env, m)
    replay_buffer.save_game(game)
    for i in range(20):
        m.train_on_batch(replay_buffer.sample_batch())
    rew = sum(game.rewards)
    rews.append(rew)
    print(len(game.history), rew, collections.Counter(game.history), m.losses[-1][0])