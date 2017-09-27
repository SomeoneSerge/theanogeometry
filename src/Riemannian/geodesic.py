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
    x = M.element()
    v = M.covector()

    def ode_geodesic(t,x):
        dx2t = - T.tensordot(T.tensordot(x[1],
                                         M.Gamma_g(x[0]), axes = [0,1]),
                             x[1],axes = [1,0])
        dx1t = x[1]
        return T.stack((dx1t,dx2t))

    geodesic = lambda x,v: integrate(ode_geodesic, T.stack((x,v)))
    M.Expt = lambda x,v: geodesic(x,v)[1][:,0]
    M.Exptf = theano.function([x,v], M.Expt(x,v))

