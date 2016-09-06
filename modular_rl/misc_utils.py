import numpy as np

# configure ==============
def update_default_config(tuples, usercfg):
    """
    tuples: a sequence of 4-tuples (name, type, defaultvalue, description)
    usercfg: dict-like object specifying overrides

    outputs
    ------
    dict2 with updated configuration
    """
    out = dict2()
    for (name, _, defval, _) in tuples:
        out[name] = defval
    if usercfg:
        pass
    return out

def comma_sep_ints(s):
    if s:
        return map(int, s.split(","))
    else:
        return []

# misc ====================
class dict2(dict):
    "dictionary like object that exposes its keys as attributes"
    def __init__(self, **kwargs):
        dict.__init__(self, kwargs)
        self.__dict__ = self
