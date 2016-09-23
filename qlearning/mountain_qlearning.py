##
import gym
import numpy as np
import random
import argparse, ConfigParser
import pandas as pd
##

class Agent(object):
    def __init__(self, num_states=100, num_actions=4, alpha=0.2, gamma=0.9, explore_rate=0.5, explore_rate_decay=0.99):
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.explore_rate = explore_rate
        self.explore_rate_decay = explore_rate_decay
        self.state = 0
        self.action = 0
        self.qtable = np.random.uniform(low=-1, high=1, size=(num_states, num_actions))

    def set_initial_state(self, state):
        """
        @summary:       set the initial state and return an action
        @param state:   initial state
        @return:        selected action
        """
        self.state = state
        # select the action with best q value
        self.action = self.qtable[state].argsort()[-1]
        return self.action

    def move(self, state_new, reward):
        """
        @summary:       move to the new state with given old state and reward
        @param state:   new state
        @param reward:  reward (old state to new state)
        @return:        next action (for the new state)
        """
        alpha = self.alpha
        gamma = self.gamma
        state = self.state
        action = self.action
        qtable = self.qtable
        
        use_random_action = np.random.uniform(0,1) < self.explore_rate
        if use_random_action:
            action_next = random.randint(0, self.num_actions-1)
        else:
            action_next = self.qtable[state_new].argsort()[-1]

        # update explore rate, qtable, state, action
        #self.explore_rate *= self.explore_rate_decay
        qtable[state, action] = (1-alpha)*qtable[state, action] + alpha*(reward + gamma*qtable[state_new, action_next])
        self.state = state_new
        self.action = action_next

        return action_next
##

if __name__ == '__main__':
    random.seed(0)
    #parser = argparse.ArgumentParser(description='run Q learning algorithm')
    #parser.add_argument('config', type=str, help='config file path')
    #args = parser.parse_args()
    #cfg_path = 'cfg/CartPole-v0.cfg'
    #cfg = ConfigParser.RawConfigParser()
    #cfg.read(cfg_path)

    # set hyper param
    num_steps = 200
    num_episodes = 50000
    render = False

    env = gym.make('MountainCar-v0')
    # cart-pole rules: cart within 2.4 units from center, pole within 15 degrees from vertical
    pos_bins = pd.cut([-1.0, 0.6], bins=10, retbins=True)[1][1:-1]
    vel_bins = pd.cut([-0.07, 0.07], bins=10, retbins=True)[1][1:-1]

    agent = Agent(num_states=10**env.observation_space.shape[0],
                  num_actions=env.action_space.n,
                  alpha = 0.1,
                  gamma = 0.99,
                  explore_rate = 0.5,
                  explore_rate_decay = 0.999)
    
    ##
    rewards_list = np.ndarray(0)
    for i in xrange(num_episodes - 1):
        print "episode %i" % i
        observation = env.reset()
        pos, vel = observation
        def build_state(features):
            return int("".join(map(lambda feature: str(int(feature)), features)))
        def to_bin(value, bins):
            return np.digitize(x=[value], bins=bins)[0]
        state = build_state([to_bin(pos, pos_bins),
                             to_bin(vel, vel_bins)])
        action = agent.set_initial_state(state)

        total_rewards = 0
        #if i > 10000: render=True
        for step in xrange(num_steps-1):
            observation, reward, done, info = env.step(action)
            total_rewards += reward
            pos, vel = observation
            state = build_state([to_bin(pos, pos_bins),
                                 to_bin(vel, vel_bins)])
            action = agent.move(state, reward)
            
            if render and step%3: env.render()
            if done:
                print "episode finished with %i steps" % (step+1)
                break

        rewards_list = np.append(rewards_list, [total_rewards])
        if len(rewards_list) > 100:
            rewards_list = np.delete(rewards_list, 0)
        print "last 100 rewards mean %i" % int(rewards_list.mean())
        if rewards_list.mean() >= -110:
            break
        print "last reward %i" % total_rewards
        print "explore_rate %f" % agent.explore_rate
        # decay exploration once per episode
        agent.explore_rate *= agent.explore_rate_decay
##
