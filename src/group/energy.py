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
    """ group Lagrangian and Hamiltonian from invariant metric """

    g = G.element() # \RR^{NxN} matrix
    hatxi = G.Vvector() # \RR^G_dim vector
    q = G.Vvector() # element in psi coordinates
    v = G.coordsvector() # \RR^G_dim tangent vector in coordinates
    vg = G.vector() # \RR^{NxN} tangent vector at g
    p = G.coordscovector() # \RR^G_dim cotangent vector in coordinates
    mu = G.Vcovector() # \RR^G_dim LA cotangent vector in coordinates

    # Lagrangian
    def Lagrangian(g,vg):
        return .5*G.gG(g,vg,vg)
    G.Lagrangian = Lagrangian
    G.Lagrangianf = theano.function([g,vg],G.Lagrangian(g,vg))
    # Lagrangian using psi map
    def Lagrangianpsi(q,v):
        return .5*G.gpsi(q,v,v)
    G.Lagrangianpsi = Lagrangianpsi
    G.dLagrangianpsidq = lambda q,v: T.grad(G.Lagrangianpsi(q,v),q)
    G.dLagrangianpsidv = lambda q,v: T.grad(G.Lagrangianpsi(q,v),v)
    # LA restricted Lagrangian
    def l(hatxi):
        return 0.5*G.gV(hatxi,hatxi)
    G.l = l
    G.dldhatxi = lambda hatxi: T.grad(G.l(hatxi),hatxi)
    G.Lagrangianpsif = theano.function([q,v],G.Lagrangianpsi(q,v))
    G.lf = theano.function([hatxi],G.l(hatxi))

    # Hamiltonian using psi map
    def Hpsi(q,p):
        return .5*G.cogpsi(q,p,p)
    G.Hpsi = Hpsi
    # LA^* restricted Hamiltonian
    def Hminus(mu):
        return .5*G.cogV(mu,mu)
    G.Hminus = Hminus
    G.dHminusdmu = lambda mu: T.grad(G.Hminus(mu),mu)
    G.Hpsif = theano.function([q,p],G.Hpsi(q,p))
    G.Hminusf = theano.function([mu],G.Hminus(mu))

    # Legendre transformation. The above Lagrangian is hyperregular
    G.FLpsi = lambda q,v: (q,G.dLagrangianpsidv(q,v))
    G.invFLpsi = lambda q,p: (q,G.cogpsi(q,p))
    def HL(q,p):
        (q,v) = invFLpsi(q,p)
        return T.dot(p,v)-L(q,v)
    G.HL = HL
    G.Fl = lambda hatxi: G.dldhatxi(hatxi)
    G.invFl = lambda mu: G.cogV(mu)
    def hl(mu):
        hatxi = invFl(mu)
        return T.dot(mu,hatxi)-l(hatxi)
    G.hl = hl
    G.FLpsif = theano.function([q,v],G.FLpsi(q,v))
    G.invFLpsif = theano.function([q,p],G.invFLpsi(q,p))
    G.Flf = theano.function([hatxi],G.Fl(hatxi))
    G.invFlf = theano.function([mu],G.invFl(mu))

    # default Hamiltonian
    G.H = G.Hpsi
    G.Hf = theano.function([q,p],G.H(q,p))

# A.set_value(np.diag([3,2,1]))
# print(FLpsif(q0,v0))
# print(invFLpsif(q0,p0))
# (flq0,flv0)=FLpsif(q0,v0)
# print(q0,v0)
# print(invFLpsif(flq0,flv0))
