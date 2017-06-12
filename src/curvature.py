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

# Curvature:
def R(x):
    return T.tensordot(Gamma_gM(x),Gamma_gM(x),axes = [0,2]).dimshuffle(0,3,1,2) - T.tensordot(Gamma_gM(x),Gamma_gM(x),axes = [0,2]).dimshuffle(3,0,1,2) + T.jacobian(Gamma_gM(x).flatten(),x).reshape((d,d,d,d)).dimshuffle(1,3,2,0) - T.jacobian(Gamma_gM(x).flatten(),x).reshape((d,d,d,d)).dimshuffle(3,1,2,0)

def R_ui(x,ui):
    return T.tensordot(T.nlinalg.matrix_inverse(ui),T.tensordot(R(x),ui,(2,0)),(1,2)).dimshuffle(1,2,0,3)

Rf = theano.function([x], R(x))
R_uif = theano.function([x,ui], R_ui(x,ui))

# Sectional Curvature:


