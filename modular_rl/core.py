from importlib import import_module
from .misc_utils import *

def get_agent_cls(name):
    p, m = name.rsplit('.', 1)
    print(p)
    print(m)
    mod = import_module(p)
    constructor = getattr(mod, m)
    return constructor

class ProbType(object):
    def sampled_variable(self):
        raise NotImplementedError
    def prob_variable(self):
        raise NotImplementedError
    def likelihood(self):
        raise NotImplementedError
    def loglikelihood(self):
        raise NotImplementedError
    def kl(self, prob0, prob1):
        raise NotImplementedError
    def entropy(self, prob):
        raise NotImplementedError
    def maxprob(self, prob):
        raise NotImplementedError

class DiagGauss(ProbType):
    def __init__(self, d):
        self.d = d
    def sampled_variable(self):
        return T.matrix('a')

class Categorical(ProbType):
    def __init__(self, n):
        self.n = n
