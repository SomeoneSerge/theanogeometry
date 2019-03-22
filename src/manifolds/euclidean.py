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

from src.manifolds.manifold import *

class Euclidean(Manifold):
    """ Euclidean space """

    def __init__(self,N=3):
        Manifold.__init__(self)
        self.dim = constant(N)

        self.g = lambda x: T.eye(self.dim)

        # action of matrix group on elements
        x = self.element()
        g = T.matrix() # group matrix
        gs = T.tensor3() # sequence of matrices
        self.act = lambda g,x: T.tensordot(g,x,(1,0))
        self.actf = theano.function([g,x], self.act(g,x))
        self.actsf = theano.function([gs,x], self.act(gs,x))

    def __str__(self):
        return "Euclidean manifold of dimension %d" % (self.dim.eval())
