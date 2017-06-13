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
from src.manifold import *
from src.metric import *

def ode_geodesic(t,x):
    
    dx2t = - T.tensordot(T.tensordot(x[1], Gamma_gM(x[0]), axes = [0,1]),
                        x[1],axes = [1,0])
    dx1t = x[1]
    
    return T.stack((dx1t,dx2t))

geo = lambda q,p: integrate(ode_geodesic, T.stack((q,p)))
Expt = lambda q,p: geo(q,p)[1][:,0]
Exptf = theano.function([q,p], Expt(q,p))
