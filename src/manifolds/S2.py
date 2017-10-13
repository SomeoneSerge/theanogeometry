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

from src.manifolds.ellipsoid import *

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.ticker as ticker

class S2(Ellipsoid):
    """ 2d Sphere """

    def __init__(self):
        Ellipsoid.__init__(self,params=[1.,1.,1.])

        # spherical coordinates
        x = self.coords() # Point on M in coordinates
        self.F_spherical = lambda phitheta: T.stack([T.sin(phitheta[1])*T.cos(phitheta[0]),T.sin(phitheta[1])*T.sin(phitheta[0]),T.cos(phitheta[1])])
        self.F_sphericalf = theano.function([x], self.F_spherical(x))
        self.JF_spherical = lambda x: T.jacobian(self.F_spherical(x),x)
        self.JF_sphericalf = theano.function([x], self.JF_spherical(x))
        self.F_spherical_inv = lambda x: T.stack([T.arctan2(x[1],x[0]),T.arccos(x[2])])
        self.F_spherical_invf = theano.function([x], self.F_spherical_inv(x))
        self.g_spherical = lambda x: T.dot(self.JF_spherical(x).T,self.JF_spherical(x))
        self.mu_Q_spherical = lambda x: 1./T.nlinalg.Det()(self.g_spherical(x))
        self.mu_Q_sphericalf = theano.function([x],self.mu_Q_spherical(x))

    def __str__(self):
        return "%dd sphere (ellipsoid parameters %s)" % (self.dim.eval(),self.params.eval())

