from efprob.efprob_qu import *
# import efprob_qu
from typing import *

foo = ket(0)
print(foo)


Pred = Callable[..., bool]
Q = TypeVar('Q', bound=Pred)


# https://stackoverflow.com/a/42561985/9075422
from functools import wraps

def negate(f:Q) -> Q:
    @wraps(f)
    def g(*args,**kwargs) -> bool:
        return not f(*args,**kwargs)
    return g

iseq = lambda x,y: x==y
print(negate(iseq)(3,4))

isEntangled = lambda s: negate(entanglement_test)(s)


# def isEntangled(arg: 'State') -> bool:
#     return not entanglement_test(arg)


print(isEntangled(foo@foo))
