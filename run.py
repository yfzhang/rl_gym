#!/usr/bin/env python

from __future__ import division
import colored_traceback.always

import tensorflow as tf
from config import parse_flags
import gym
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.memory import SequentialMemory, EpisodeParameterMemory
from rl.policy import BoltzmannQPolicy
from rl.core import Agent
from rl.agents.dqn import DQNAgent
from rl.agents.cem import CEMAgent
from rl.callbacks import FileLogger, ModelIntervalCheckpoint

FLAGS = parse_flags()


def make_env():
    env = gym.make(FLAGS.game)
    FLAGS.num_actions = env.action_space.n
    return env


def make_model():
    if FLAGS.model == "3dense":
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
    elif FLAGS.model == "no_hidden":
        model = Sequential()
        model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
        model.add(Dense(FLAGS.num_actions))
        model.add(Activation('softmax'))
    else:
        raise BaseException("unknown model, please check your flag [--model]")
    print(model.summary())
    return model

def make_agent():
    policy = BoltzmannQPolicy()
    model = make_model()
    if FLAGS.agent == "dqn":
        memory = SequentialMemory(limit=50000, window_length=1)
        agent = DQNAgent(model=model, nb_actions=FLAGS.num_actions, memory=memory, nb_steps_warmup=10,
                         target_model_update=1e-2, policy=policy)
        agent.compile(Adam(lr=1e-3), metrics=['mae'])
    elif FLAGS.agent == "cem":
        memory = EpisodeParameterMemory(limit=1000, window_length=1)
        agent = CEMAgent(model=model, nb_actions=FLAGS.num_actions, memory=memory, batch_size=100, nb_steps_warmup=10,
                         train_interval=50, elite_frac=0.05)
        agent.compile()
    else:
        raise BaseException("unknown agent, please check your flag [--agent]")
    return agent

env = make_env()
agent = make_agent()
if FLAGS.mode == "train":
    weights_filename = FLAGS.exp_dir + 'weights.h5f'
    checkpoint_weights_filename = FLAGS.exp_dir + 'weights_{step}.h5f'
    log_filename = FLAGS.exp_dir + 'log.json'
    callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=FLAGS.save_weight_interval)]
    callbacks += [FileLogger(log_filename, interval=FLAGS.save_log_interval)]
    agent.fit(env, callbacks=callbacks, nb_steps=FLAGS.max_steps, visualize=FLAGS.visualize_train, verbose=2)
    agent.test(env, nb_episodes=5, visualize=True)
elif FLAGS.mode == "test":
    pass
else:
    raise BaseException("unknown mode, please check your flag [--mode]")
