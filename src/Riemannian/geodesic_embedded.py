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

from src.manifolds.manifold import EmbeddedManifold

def initialize(M):
    assert (isinstance(M, EmbeddedManifold))

    x = M.element()
    v = M.covector()

    # Euclidean 'rotation' of vector into tangent space of X
    def PT(x, Jv):
        # (w,V) = linalg.symEigh()(T.dot(M.JF(x),M.JF(x).T))
        (_, V) = linalg.symEighSqrt()(M.JF(x))
        # V = V[:M.dim,:]
        u = T.tensordot(V, Jv, (0, 0))
        return T.nlinalg.norm(Jv, 2) * u / M.norm(x, u)

    " approximate scheme from arXiv:1711.08014 "
    def ode_geodesic(t, x):
        dx2t = PT(x[0] + dt * x[1], T.dot(M.JF(x[0]), x[1])) - x[1]
        dx1t = x[1]
        return T.stack((dx1t, dx2t))

    geodesic = lambda x, v: integrate(ode_geodesic, T.stack((x, v)))
    M.Exp_embedded = lambda x, v: geodesic(x, v)[1][-1, 0]
    M.Exp_embeddedf = theano.function([x, v], M.Exp_embedded(x, v))
    M.Exp_embeddedt = lambda x, v: geodesic(x, v)[1][:, 0]
    M.Exp_embeddedtf = theano.function([x, v], M.Exp_embeddedt(x, v))

