# From: https://gist.github.com/mrluanma/1480728
flatten = lambda lst: reduce(lambda l, i: l + flatten(i) if isinstance(i, (list, tuple)) else l + [i], lst, [])


import matplotlib.pyplot as plt
from itertools import *
import operator
from toolz import *
from toolz.curried import *

from efprob.efprob_qu import *

import sympy as sym
from sympy.interactive.printing import init_printing

# !pip install more-itertools
from more_itertools import *

def R(theta, dim=2):
    """
    Rotation array.
    ----
    theta: a scalar or a list.
    dim: 2 (default) or 3

    return: the rotation array(s) of the shape (dim, dim).
    """

    if (type(theta) != list): theta = [theta]

    rs = list()
    for t in theta:
        r = np.array([[np.cos(t), -np.sin(t)],
                      [np.sin(t), np.cos(t)]])
        if dim==3:
            r = np.pad(r, ((0,1),(0,1)), 'constant')
            r[-1, -1] = 1
        rs.append(r)
    if len(rs)==1: rs = rs[0]
    return rs

def sqrt_diag(E):
    if type(E) != list: E = [E]
    v, s, vh = np.linalg.svd(E)
    Ms = [v.dot(np.diag(np.sqrt(s)).dot(vh)) for v,s,vh in zip(v,s,vh)]
    if len(Ms) == 1: Ms = Ms[0]
    return Ms

# def prints(iterator, func= lambda arg: arg):
#     for x in iterator:
#         print(func(x))

def prints(*iterator, func= lambda arg: arg):
    for x in iterator:
        print(func(x))
