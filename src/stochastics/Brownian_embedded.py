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
    """ Brownian motion using embedded space approximation for geodesic steps """
    assert (isinstance(M, EmbeddedManifold))
    assert (hasattr(M, 'invF'))

    x = M.element()
    dW = M.element()
    t = T.scalar()

    def sde_Brownian_embedded(dW, t, x):
        gsharpx = M.gsharp(x)
        X = theano.tensor.slinalg.Cholesky()(gsharpx)
        det = T.zeros_like(x)
        sto = M.invF(M.F(x) + T.tensordot(M.JF(x), dW, (1, 0))) - x
        return (det, sto, X)

    M.sde_Brownian_embedded = sde_Brownian_embedded
    M.sde_Brownian_embeddedf = theano.function([dW, t, x], M.sde_Brownian_embedded(dW, t, x), on_unused_input='ignore')
    M.Brownian_embedded = lambda x, dWt: integrate_sde(sde_Brownian_embedded, integrator_ito, x, dWt)
    M.Brownian_embeddedf = theano.function([x, dWt], M.Brownian_embedded(x, dWt))
