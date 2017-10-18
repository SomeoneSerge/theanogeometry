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

    xs = T.matrix() # point to be advected with flow
    qt = T.matrix() # fixed flow, q part
    pt = T.matrix() # fixed flow, p part

    def ode_Ham_advect(q,p,t,x):
        dxt = T.tensordot(M.K(x,q.reshape((-1,M.m))),p,(1,0)).reshape((-1,M.m))
        return dxt

    M.Ham_advect = lambda xs,qt,pt: integrate(ode_Ham_advect,xs,qt,pt)
    M.Ham_advectf = theano.function([xs,qt,pt], M.Ham_advect(xs,qt,pt))
