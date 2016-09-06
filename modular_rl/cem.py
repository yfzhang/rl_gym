import os
import numpy as np
from . import parallel_utils
from .core import *

def cem(f, th_mean, batch_size, n_iter, elite_frac, initial_std=1.0, extra_std=0.0, std_decay_time=1.0, pool=None):
    """
    noisy cross entropy method

    f: function of one argument--the parameter vector
    th_mean: initial distribution is theta ~ N(th_mean, initial_std)
    batch_size: the number of samples of theta per iteration
    n_iter: the number of iterations
    elite_frac: the fraction of samples to select as at the end of each iteration, and use for fitting new distribution
    initial_std: standard deviation of initial distribution
    extra_std: "noise" component added to increase standard deviation
    std_decay_time: the number of timesteps it takes for noise to decay
    """

    n_elite = int(np.round(batch_size*elite_frac))
    th_std = np.ones(th_mean.size)*initial_std

    for iteration in xrange(n_iter):
        extra_var_multiplier = max((1.0-iteration/float(std_decay_time)), 0)
        sample_std = np.sqrt(th_std + np.square(extra_std) * extra_var_multiplier)

        ths = np.array([th_mean + dth for dth in sample_std[None,:]*np.random.randn(batch_size, th_mean.size)])
        if pool is None: ys = np.array(map(f, ths))
        else: ys = np.array(pool.map(f, ths))
        assert ys.ndim == 1
        elite_inds = ys.argsort()[-n_elite:]
        elite_ths = ths[elite_inds]
        th_mean = elite_ths.mean(axis=0)
        th_std = elite_ths.var(axis=0)
        yield {"ys":ys, "th":th_mean, "ymean":ys.mean(), "std":sample_std}

CEM_OPTIONS = [
    ("batch_size", int, 200, "the number of episodes per batch"),
    ("n_iter", int, 200, "the number of iterations"),
    ("elite_frac", float, 0.2, "fraction of samples used to fit new distribution"),
    ("initial_std", float, 1.0, "initial standard deviation for parameters"),
    ("extra_std", float, 0.0, "extra stddev added"),
    ("std_decay_time", float, -1.0, "the number of timesteps that extra decays over. negative => n_iter/2"),
    ("timestep_limit", int, 0, "maximum length of trajectories"),
    ("parallel", int, 0, "collect trajectories in parallel"),
]

def run_cem_algorithm(env, agent, usercfg=None, callback=None):
    cfg = update_default_config(CEM_OPTIONS, usercfg)
    print "cem cfg", cfg

    G = parallel_utils.G
    G.env = env
    G.agent = agent
    G.timestep_limit = cfg["timestep_limit"]

    # TODO: add parallel implementation
    pool = None

    th_mean = agent.get_flat()
    
