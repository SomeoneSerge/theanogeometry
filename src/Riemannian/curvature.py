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
    """ Riemannian curvature tensor """

    d = M.dim
    x = M.element()
    u = M.frame()
    e1 = M.vector()
    e2 = M.vector()

    def R(x):
        return T.tensordot(M.Gamma_g(x),M.Gamma_g(x),axes = [0,2]).dimshuffle(0,3,1,2) - T.tensordot(M.Gamma_g(x),M.Gamma_g(x),axes = [0,2]).dimshuffle(3,0,1,2) + T.jacobian(M.Gamma_g(x).flatten(),x).reshape((d,d,d,d)).dimshuffle(1,3,2,0) - T.jacobian(M.Gamma_g(x).flatten(),x).reshape((d,d,d,d)).dimshuffle(3,1,2,0)

    def R_u(x,u):
        return T.tensordot(T.nlinalg.matrix_inverse(u),T.tensordot(R(x),u,(2,0)),(1,2)).dimshuffle(1,2,0,3)

    M.R = R
    M.Rf = theano.function([x], R(x))
    M.R_u = R_u
    M.R_uf = theano.function([x,u], R_u(x,u))

    # Sectional Curvature:
    def sec_curv(x,e1,e2):
        Rm = T.tensordot(M.g(x),M.R(x), [1,0])
        sec = T.tensordot(T.tensordot(T.tensordot(T.tensordot(Rm, e1, [0,0]), 
                                                  e2, [0,0]),
                          e2, [0,0]), e1, [0,0])
        return sec

    M.sec_curv = sec_curv
    M.sec_curvf = theano.function([x,e1,e2],sec_curv(x,e1,e2))


