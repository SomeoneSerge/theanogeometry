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

from src.group import *
from src.utils import *
from src.metric import *

#######################################################################
# general Brownian motion with respect to left/right invariant metric #
#######################################################################

assert(invariance == 'left')

def sde_Brownian_inv(dW,t,g):
    X = T.tensordot(invpf(g,eiLA),sigma,(2,0))
    det = -.5*T.tensordot(T.diagonal(C,0,2).sum(1),X,(0,2))
    sto = T.tensordot(X,dW,(2,0))
    return (det,sto,X)
Brownian_inv = lambda g,dWt: integrate_sde(sde_Brownian_inv,integrator_stratonovich,g,dWt)
Brownian_invf = theano.function([g,dWt], Brownian_inv(g,dWt))

