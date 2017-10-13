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
from src.manifold import *
from src.FM import *

# Development and Stochastic Development
dgamma = T.matrix()
dW = T.matrix()

## Development: (Deterministic)
def ode_Dev(dgamma,t,q):
    
    x = q[0:d]
    ui = q[d:(d+rank*d)].reshape((d,rank))

    det = T.tensordot(Hori(x,ui), dgamma, axes = [1,0])
    
    return det

dev = lambda q,dgamma: integrate(ode_Dev,q,dgamma)[1]
devf = theano.function([q,dgamma], dev(q,dgamma))

# Stochastic Development:
def sde_SD(dWt,t,q,drift):
    
    x = q[0:d]
    ui = q[d:(d+rank*d)].reshape((d,rank))

    det = T.tensordot(Hori(x,ui), drift, axes = [1,0]) # T.zeros_like(q) 
    sto = T.tensordot(Hori(x,ui), dWt, axes = [1,0])
    
    return (det, sto, T.constant(0.), T.constant(0.))

stoc_dev = lambda q,dWt,drift: integrate_sde(sde_SD,integrator_stratonovich,q,dWt,drift)[1]
stoc_devf = theano.function([q,dWt,drift], stoc_dev(q,dWt,drift))

# TO DO: Incorporate drift!!
