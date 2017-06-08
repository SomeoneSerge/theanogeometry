from src.setup import *
from src.params import *

import importlib
globals().update(importlib.import_module('src.manifolds.'+manifold).__dict__)
print("Manifold: ", manifold)

q = T.vector() # Point on M in coordinates
q1 = T.vector() # Point on M in coordinates
zeroU = T.zeros((d,)) # zero element in coordinates U
X = T.vector() # Frame vector of T_qM
p = T.vector() # Covector of T_qM.
qp = T.matrix() # Matrix of p and q.
#Method = T.fscalar()
x = T.vector() # Point on M
ui = T.matrix() # Frame of T_xM
dW = T.matrix() # Process in R^n
drift = T.vector() # Drift of stochastic process
gamma = T.matrix() # curve in R^n

