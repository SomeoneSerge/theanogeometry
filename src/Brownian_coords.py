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

from src.manifold import *
from src.utils import *
from src.metric import *

#######################################################################
# general Brownian motion in coodinates                               #
#######################################################################

def sde_Brownian_coords(dW,t,q):
    gMsharpq = gMsharp(q)
    X = theano.tensor.slinalg.Cholesky()(gMsharpq)
    det = T.tensordot(gMsharpq,Gamma_gM(q),((0,1),(0,1)))
    sto = T.tensordot(X,dW,(1,0))
    return (det,sto,X)
Brownian_coords = lambda x,dWt: integrate_sde(sde_Brownian_coords,integrator_ito,x,dWt)
Brownian_coordsf = theano.function([q,dWt], Brownian_coords(q,dWt))

