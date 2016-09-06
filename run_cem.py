#!/usr/bin/env python
"""
this script runs the cross entropy method
"""

import gym, logging, argparse, sys
from modular_rl import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--env", required=True)
    parser.add_argument("--agent", required=True)
    args = parser.parse_args()

    env = gym.make(args.env)
    agent_ctor = get_agent_cls(args.agent)
    cfg = args.__dict__
    agent = agent_ctor(env.observation_space, env.action_space, cfg)
    #print(agent)
    #print(cfg)

    COUNTER = 0
    run_cem_algorithm(env, agent)
