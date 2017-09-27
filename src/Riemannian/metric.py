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
    """ add metric related structures to manifold """

    d = M.dim
    x = M.element()

    if hasattr(M, 'g'):
        if not hasattr(M, 'gsharp'):
            M.gsharp = lambda x: T.nlinalg.matrix_inverse(M.g(x))
            M.gsharpf = theano.function([x],M.gsharp(x))
    elif hasattr(M, 'gsharp'):
        if not hasattr(M, 'g'):
            M.g = lambda x: T.nlinalg.matrix_inverse(M.gsharp(x))
            M.gf = theano.function([x],M.g(x))
    else:
        raise ValueError('no metric or cometric defined on manifold')


    M.Dg = lambda x: T.jacobian(M.g(x).flatten(),x).reshape((d,d,d)) # Derivative of metric
    M.Dgf = theano.function([x],M.Dg(x))

    ##### Measure
    M.mu_Q = lambda x: 1./T.nlinalg.Det()(M.g(x))
    M.mu_Qf = theano.function([x],M.mu_Q(x))

    ##### Sharp and flat map:
#    M.Dgsharp = lambda q: T.jacobian(M.gsharp(q).flatten(),q).reshape((d,d,d)) # Derivative of sharp map
    v = M.vector()
    p = M.covector()
    M.flat = lambda x,v: T.dot(M.g(x),v)
    M.flatf = theano.function([x,v], M.flat(x,v))
    M.sharp = lambda x,p: T.dot(M.gsharp(x),p)
    M.sharpf = theano.function([x,p], M.sharp(x,p))

    ##### Christoffel symbols
    M.Gamma_g = lambda x: 0.5*(T.tensordot(M.gsharp(x),M.Dg(x),axes = [1,0])
                   +T.tensordot(M.gsharp(x),M.Dg(x),axes = [1,0]).dimshuffle(0,2,1)
                   -T.tensordot(M.gsharp(x),M.Dg(x),axes = [1,2]))
    M.Gamma_gf = theano.function([x],M.Gamma_g(x))

    # Inner Product from g
    w = M.vector()
    M.dot = lambda x,v,w: T.dot(T.dot(M.g(x),w),v)
    M.dotf = theano.function([x,v,w],M.dot(x,v,w))
    M.norm = lambda x,v: T.sqrt(M.dot(x,v,v))
    M.normf = theano.function([x,v],M.norm(x,v))
    pp = M.covector()
    M.dotsharp = lambda x,p,pp: T.dot(T.dot(M.gsharp(x),pp),p)
    M.dotsharpf = theano.function([x,p,pp],M.dotsharp(x,pp,p))

    #def GramSchmidt(u,q):
    #    return (GramSchmidt_f(innerProd))(u,q)

    ##### Hamiltonian
    q = M.element()
    p = M.covector()
    M.H = lambda q,p: 0.5*T.dot(p,T.dot(M.gsharp(q),p))
    M.Hf = theano.function([q,p],M.H(q,p))

