"""
"agent" here is a container with the policy, value function, etc.
this file implements a bunch of agents
"""

from modular_rl import *
from gym.spaces import Box, Discrete
from keras.models import Sequential
from keras.layers.core import Dense

MLP_OPTIONS = [
    ("hid_sizes", comma_sep_ints, [64, 64], "size of hidden layers of MLP"),
    ("activation", str, "tanh", "nonlinearity"),
]

def make_deterministic_mlp(ob_space, ac_space, cfg):
    assert isinstance(ob_space, Box)
    hid_sizes = cfg["hid_sizes"]
    if isinstance(ac_space, Box):
        outdim = ac_space.shape[0]
        probtype = DiagGauss(outdim)
    elif isinstance(ac_space, Discrete):
        outdim = ac_space.n
        probtype = Categorical(outdim)
    
    net = Sequential()
    for (i, layeroutsize) in enumerate(hid_sizes):
        inshp = dict(input_shape=ob_space.shape) if i==0 else {}
        net.add(Dense(layeroutsize, activation="tanh", **inshp))
    inshp = dict(input_shape=ob_space.shape) if len(hid_sizes) == 0 else {}
    net.add(Dense(outdim,  **inshp))
    Wlast = net.layers[-1].W
    print(type(Wlast))
    Wlast.set_value(Wlast.get_value(borrow=True)*0.1)
    policy = StochPolicyKeras(net, probtype)
    return policy

def make_filters(cfg, ob_space):
    if cfg["filter"]:
        obfilter = ZFilter(ob_space.shape, clip=5)
        rewfilter = ZFilter((), demean=False, clip=10)
    else:
        obfilter = IDENTITY
        rewfilter = IDENTITY
    return obfilter, rewfilter

FILTER_OPTIONS = [
    ("filter", int, 1, "whether to do a running average filter of the incoming observations and rewards"),
]

class AgentWithPolicy(object):
    def __init__(self, policy, obfilter, rewfilter):
        self.policy = policy
        self.obfilter = obfilter
        self.rewfilter = rewfilter
        self.stochastic = True
    def set_stochastic(self, stochastic):
        self.stochastic = stochastic
    def act(self, ob_no):
        return self.policy.act(ob_no, stochastic = self.stochastic)
    def get_flat(self):
        return self.policy.get_flat()
    def set_from_flat(self, th):
        return self.policy.set_from_flat(th)
    def obfilt(self, ob):
        return self.obfilter(ob)
    def rewfilt(self, rew):
        return self.rewfilter(rew)

class DeterministicAgent(AgentWithPolicy):
    options = MLP_OPTIONS + FILTER_OPTIONS
    def __init__(self, ob_space, ac_space, usercfg):
        cfg = update_default_config(self.options, usercfg)
        policy = make_deterministic_mlp(ob_space, ac_space, cfg)
        obfilter, rewfilter = make_filters(cfg, ob_space)
        AgentWithPolicy.__init__(self, policy, obfilter, rewfilter)
        self.set_stochastic(False)
