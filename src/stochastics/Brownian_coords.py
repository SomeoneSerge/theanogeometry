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
from src.utils import *

def initialize(M):
    """ Brownian motion in coordinates """

    x = M.element()
    dW = M.element()
    t = T.scalar()

    def sde_Brownian_coords(dW,t,q):
        gsharpq = M.gsharp(q)
        X = theano.tensor.slinalg.Cholesky()(gsharpq)
        det = T.tensordot(gsharpq,M.Gamma_g(q),((0,1),(0,1)))
        sto = T.tensordot(X,dW,(1,0))
        return (det,sto,X)
    M.sde_Brownian_coords = sde_Brownian_coords
    M.sde_Brownian_coordsf = theano.function([dW,t,x], M.sde_Brownian_coords(dW,t,x), on_unused_input = 'ignore') 
    M.Brownian_coords = lambda x,dWt: integrate_sde(sde_Brownian_coords,integrator_ito,x,dWt)
    M.Brownian_coordsf = theano.function([x,dWt], M.Brownian_coords(x,dWt))

