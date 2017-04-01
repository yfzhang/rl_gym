#!/usr/bin/env python

import tensorflow as tf
from config import parse_flags
import gym
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.memory import SequentialMemory
from rl.policy import BoltzmannQPolicy
from rl.agents.dqn import DQNAgent


FLAGS = parse_flags()


def make_env():
    env = gym.make(FLAGS.game)
    FLAGS.num_actions = env.action_space.n
    return env


def make_model():
    model = Sequential()
    model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(FLAGS.num_actions))
    model.add(Activation('linear'))
    print(model.summary())
    return model


def make_memory():
    memory = SequentialMemory(limit=50000, window_length=1)
    return memory


def make_agent():
    policy = BoltzmannQPolicy()
    memory = make_memory()
    model = make_model()
    agent = DQNAgent(model=model, nb_actions=FLAGS.num_actions, memory=memory, nb_steps_warmup=10,
                     target_model_update=1e-2, policy=policy)
    agent.compile(Adam(lr=1e-3), metrics=['mae'])
    return agent


env = make_env()
agent = make_agent()
agent.fit(env, nb_steps=50000, visualize=True, verbose=2)
agent.test(env, nb_episodes=5, visualize=True)
