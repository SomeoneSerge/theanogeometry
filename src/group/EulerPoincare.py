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

def initialize(G):
    """ Euler-Poincare geodesic integration """

    assert(G.invariance == 'left')

    g = G.element() # \RR^{NxN} matrix
    q = G.Vvector() # element in psi coordinates
    v = G.coordsvector() # \RR^G_dim tangent vector in coordinates
    mu = G.Vcovector() # \RR^G_dim LA cotangent vector in coordinates

    def ode_EP(t,mu):
        xi = G.invFl(mu)
        dmut = -G.coad(xi,mu)
        return dmut
    G.EP = lambda mu: integrate(ode_EP,mu)
    G.EPf = theano.function([mu], G.EP(mu))

    # reconstruction
    def ode_EPrec(mu,t,g):
        xi = G.invFl(mu)
        dgt = G.dL(g,G.e,G.VtoLA(xi))
        return dgt
    G.EPrec = lambda g,mus: integrate(ode_EPrec,g,mus)
    mus = T.matrix() # mu for each time step
    G.EPrecf = theano.function([g,mus], G.EPrec(g,mus))

    ### geodesics
    G.coExpEP = lambda g,mu: G.EPrec(g,G.EP(mu)[1])[1][-1]
    G.ExpEP = lambda g,v: G.coExpEP(g,G.flatV(v))
    G.ExpEPpsi = lambda q,v: G.ExpEP(G.psi(q),G.flatV(v))
    G.coExpEPt = lambda g,mu: G.EPrec(g,G.EP(mu)[1])
    G.ExpEPt = lambda g,v: G.coExpEPt(g,G.flatV(v))
    G.ExpEPpsit = lambda q,v: G.ExpEPt(G.psi(q),G.flatV(v))
    G.DcoExpEP = lambda g,mu: (
        T.jacobian(G.coExpEP(g,mu).flatten(),g).reshape(G.N,G.N,G.N,G.N),
        T.jacobian(G.coExpEP(g,mu).flatten(),mu).reshape(G.N,G.N,G.dim)
        )
    G.ExpEPf = theano.function([g,v], G.ExpEP(g,v))
    G.ExpEPpsif = theano.function([q,v], G.ExpEPpsi(q,v))
    G.ExpEPtf = theano.function([g,v], G.ExpEPt(g,v))
    G.ExpEPpsitf = theano.function([q,v], G.ExpEPpsit(q,v))
    G.coExpEPf = theano.function([g,mu], G.coExpEP(g,mu))
    G.coExpEPtf = theano.function([g,mu], G.coExpEPt(g,mu))
    #loss = 1./G_emb_dim*T.sum(T.sqr(Exp(g,mu)-h))
    #dloss = (T.grad(loss,g),T.grad(loss,g))
    #lossf = theano.function([g,mu,h], loss)
    #dlossf = theano.function([g,mu,h], [loss, dloss[0], dloss[1]])
