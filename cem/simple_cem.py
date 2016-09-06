import gym
import numpy as np
import logging
from _policies import BinaryActionLinearPolicy
from _policies import ContinuousActionLinearPolicy
import sys, os, json
import cPickle as pickle

def cem(f, th_mean, batch_size, n_iter, elite_frac, initial_std=1.0):
    """
    generic implementation of the cross entropy method for maximizing a black-box function

    f: a function mapping from vector -> scalar
    th_mean: initial mean over input distribution
    batch_size: number of samples of theta to evaluate per batch
    n_iter: number of batches
    elite_frac: each batch, select this fraction of the top-performing samples
    initial_std: initial standard deviation over parameter vectors
    """

    n_elite = int(np.round(batch_size*elite_frac))
    th_std = np.ones_like(th_mean) * initial_std

    for _ in range(n_iter):
        ths = np.array([th_mean + dth for dth in th_std[None, :]*np.random.randn(batch_size, th_mean.size)])
        print(ths.shape)
        ys = np.array([f(th) for th in ths])
        print(ys)
        elite_inds = ys.argsort()[::-1][:n_elite]
        print(elite_inds)
        elite_ths = ths[elite_inds]
        print(elite_ths)
        print(ths.shape)
        th_mean = elite_ths.mean(axis=0)
        th_std = elite_ths.std(axis=0)
        yield {'ys' : ys, 'theta_mean' : th_mean, 'y_mean' : ys.mean()}

def do_rollout(agent, env, num_steps, render=False):
    total_rew = 0
    ob = env.reset()
    for t in range(num_steps):
        a = agent.act(ob)
        (ob, reward, done, _info) = env.step(a)
        total_rew += reward
        if render and t%3==0: env.render()
        if done: break
    return total_rew, t+1

def noisy_evaluation(theta):
    agent = BinaryActionLinearPolicy(theta)
    #agent = ContinuousActionLinearPolicy(theta)
    rew, T = do_rollout(agent, env, num_steps)
    return rew

def writefile(fname, s):
    with open(os.path.join(outdir, fname), 'w') as fh: fh.write(s)

if __name__ == '__main__':
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    env = gym.make('CartPole-v0')
    #env = gym.make('Pendulum-v0')
    env.seed(0)
    np.random.seed(0)
    params = dict(n_iter=1, batch_size=25, elite_frac=0.2)
    num_steps = 200

    # specify the directory where to write to
    outdir = './cem_agent_results'
    env.monitor.start(outdir, force=True)

    info = {}
    info['params'] = params
    info['argv'] = sys.argv
    info['env_id'] = env.spec.id

    # train the agent, and snapshot each stage
    for (i, iterdata) in enumerate(cem(noisy_evaluation, np.zeros(env.observation_space.shape[0]+1), **params)):
        print('iteration %2i. episode mean reward: %7.3f'%(i, iterdata['y_mean']))
        agent = BinaryActionLinearPolicy(iterdata['theta_mean'])
        do_rollout(agent, env, num_steps, render=True)
        writefile('agent-%.4i.pkl'%i, str(pickle.dumps(agent, -1)))

    # write out the env at the end so we store the parameters of this environment
    writefile('info.json', json.dumps(info))
    env.monitor.close()
    logger.info("successfully ran cross-entropy method")

