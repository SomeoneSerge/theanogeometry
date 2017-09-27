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

class Manifold(object):
    """ Base manifold class """

    def __init__(self):
        self.dim = None

    def element(self):
        """ return symbolic element in manifold """
        return T.vector()

    def elements(self):
        """ return symbolic sequence of elements in manifold """
        return T.matrix()

    def coords(self):
        """ return symbolic coordinate representation of point in manifold """
        return T.vector()

    def vector(self):
        """ return symbolic tangent vector """
        return T.vector()

    def vectors(self):
        """ return symbolic sequence of tangent vector """
        return T.matrix()

    def covector(self):
        """ return symbolic cotangent vector """
        return T.vector()

    def coordsvector(self):
        """ return symbolic tangent vector in coordinate representation """
        return T.vector()

    def coordscovector(self):
        """ return symbolic cotangent vector in coordinate representation """
        return T.vector()

    def frame(self):
        """ return symbolic frame for tangent space """
        return T.matrix()

    def __str__(self):
        return "abstract manifold"

class EmbeddedManifold(Manifold):
    """ Embedded manifold base class """

    def __init__(self):
        Manifold.__init__(self)
        self.emb_dim = None

    def __str__(self):
        return "abstract embedded manifold"
