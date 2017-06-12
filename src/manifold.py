# # This file is part of Theano Geometry
#
# Copyright (C) 2017, Stefan Sommer (sommer@di.ku.dk)
# https://bitbucket.org/stefansommer/theanogemetry
#
# Theano Geometry is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Theano Geometry is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Theano Geometry. If not, see <http://www.gnu.org/licenses/>.
#

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

