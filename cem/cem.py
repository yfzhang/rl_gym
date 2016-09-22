##
import numpy as np
import gym
from gym.spaces import Discrete, Box
import argparse, ConfigParser
##
class DeterministicDiscreteActionLinearPolicy(object):
    def __init__(self, theta, ob_space, ac_space):
        """
        dim_ob: dimension of observations
        n_actions: number of actions
        theta: flat vector of parameters
        """
        dim_ob = ob_space.shape[0]
        n_actions = ac_space.n
        assert len(theta) == (dim_ob + 1) * n_actions
        self.W = theta[0 : dim_ob * n_actions].reshape(dim_ob, n_actions)
        self.b = theta[dim_ob * n_actions : None].reshape(1, n_actions)

    def act(self, ob):
        """
        """
        y = ob.dot(self.W) + self.b
        a = y.argmax()
        return a

class DeterministicContinuousActionLinearPolicy(object):
    def __init__(self, theta, ob_space, ac_space):
        """
        dim_ob: dimension of observations
        dim_ac: dimension of action vector
        theta: flat vector of parameters
        """
        self.ac_space = ac_space
        dim_ob = ob_space.shape[0]
        dim_ac = ac_space.shape[0]
        assert len(theta) == (dim_ob + 1) * dim_ac
        self.W = theta[0 : dim_ob * dim_ac].reshape(dim_ob, dim_ac)
        self.b = theta[dim_ob * dim_ac : None]
    
    def act(self, ob):
        a = np.clip(ob.dot(self.W) + self.b, self.ac_space.low, self.ac_space.high)
        return a

def do_episode(policy, env, num_steps, render=False):
    total_rew = 0
    ob = env.reset()
    for t in range(num_steps):
        a = policy.act(ob)
        (ob, reward, done, _info) = env.step(a)
        total_rew += reward
        if render and t%3==0: 
            env.render()
        if done: 
            print "episode finished after %i steps" %(t+1)
            break
    return total_rew

def noisy_evaluation(theta):
    policy = make_policy(theta)
    rew = do_episode(policy, env, num_steps)
    return rew

def make_policy(theta):
    if isinstance(env.action_space, Discrete):
        return DeterministicDiscreteActionLinearPolicy(theta, env.observation_space, env.action_space)
    elif isinstance(env.action_space, Box):
        return DeterministicContinuousActionLinearPolicy(theta, env.observation_space, env.action_space)
    else:
        raise NotImplementedError

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='run CEM algorithm')
    parser.add_argument('config', type=str, help='config file path')
    args = parser.parse_args()
    cfg_path = 'cfg/CartPole-v0.cfg'
    cfg = ConfigParser.RawConfigParser()
    cfg.read(cfg_path)
    
    ## task settings:
    env = gym.make(cfg.get('settings', 'env'))
    #env = gym.make('CartPole-v0')
    #env = gym.make('MountainCar-v0')
    #env = gym.make('Pendulum-v0')
    num_steps = cfg.getint('params', 'num_steps') # max length of epsoide

    # algorithm settings:
    n_episode = cfg.getint('params', 'num_episodes') # number of iterations of CEM
    batch_size = cfg.getint('params', 'batch_size') # number of samples per batch
    elite_frac = cfg.getfloat('params', 'elite_frac') # fraction of samples used as elite set

    if isinstance(env.action_space, Discrete):
        dim_theta = (env.observation_space.shape[0] + 1) * env.action_space.n
        print "dim_theta %i"%dim_theta
    elif isinstance(env.action_space, Box):
        dim_theta = (env.observation_space.shape[0] + 1) * env.action_space.shape[0]
    else:
        raise NotImplementedError

    # initialize mean and standard deviation
    theta_mean = np.zeros(dim_theta)
    theta_std = np.ones(dim_theta)
    ##

    #env.monitor.start('/tmp')
    for i in xrange(n_episode):
        """
        implementation code
        """
        # sample parameter vectors
        thetas = np.array([theta_mean + dth for dth in theta_std[None, :]*np.random.randn(batch_size, theta_mean.size)])
        rewards = [noisy_evaluation(theta) for theta in thetas]

        # get elite parameters
        n_elite = int(batch_size * elite_frac)
        elite_inds = np.argsort(rewards)[batch_size-n_elite:batch_size]
        elite_thetas = thetas[elite_inds]
        # update theta_mean, theta_std
        theta_mean = elite_thetas.mean(axis=0)
        theta_std = elite_thetas.std(axis=0)
        print "episode %i. mean f: %8.3g. max f:%8.3g"%(i, np.mean(rewards), np.max(rewards))
        do_episode(make_policy(theta_mean), env, num_steps, render=True)
    #env.monitor.close()
