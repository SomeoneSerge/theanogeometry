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

#############################
# Setup for S2              #
#############################

from src.manifolds.ellipsoid import *

manifold = 'S2'
ellipsoid.set_value(np.array([1,1,1]))

# spherical coordinates
q = T.vector() # Point on M in coordinates
x = T.vector() # Point on M
F_spherical = lambda phitheta: T.stack([T.sin(phitheta[1])*T.cos(phitheta[0]),T.sin(phitheta[1])*T.sin(phitheta[0]),T.cos(phitheta[1])])
F_sphericalf = theano.function([q], F_spherical(q))
JF_spherical = lambda q: T.jacobian(F_spherical(q),q)
JF_sphericalf = theano.function([q], JF_spherical(q))
F_spherical_inv = lambda x: T.stack([T.arctan2(x[1],x[0]),T.arccos(x[2])])
F_spherical_invf = theano.function([x], F_spherical_inv(x))
gM_spherical = lambda q: T.dot(JF_spherical(q).T,JF_spherical(q))
muM_Q_spherical = lambda q: 1./T.nlinalg.Det()(gM_spherical(q))
muM_Q_sphericalf = theano.function([q],muM_Q_spherical(q))
