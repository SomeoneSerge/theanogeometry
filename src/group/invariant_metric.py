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
    """ add left-/right-invariant metric related structures to group """

    g = G.element()
    hatxi = G.Vvector() # \RR^G_dim vector
    vg = G.vector() # \RR^{NxN} tangent vector at g
    wg = G.vector() # \RR^{NxN} tangent vector at g
    pg = G.covector() # \RR^{NxN} cotangent vector at g
    xiv = G.LAvector()
    xiw = G.LAvector()
    w = G.coordsvector() # \RR^G_dim tangent vector in coordinates
    v = G.coordsvector() # \RR^G_dim tangent vector in coordinates
    mu = G.Vcovector() # \RR^G_dim LA cotangent vector in coordinates

    G.sigma = theano.shared(np.eye(G.dim.eval(),G.dim.eval())) # square root cometric / diffusion field
    G.sqrtA = G.inv(G.sigma) # square root metric
    G.A = T.tensordot(G.sqrtA,G.sqrtA,(0,0)) # metric
    G.W = G.inv(G.A) # covariance (cometric)
    def gV(v=None,w=None):
        if not v and not w:
            return G.A
        elif v and not w:
            return T.tensordot(G.A,v,(1,0))
        elif v.type == T.vector().type and w.type == T.vector().type:
            return T.dot(v,T.dot(G.A,w))
        elif v.type == T.vector().type and not w:
            return T.dot(G.A,v)
        elif v.type == T.matrix().type and w.type == T.matrix().type:
            return T.tensordot(v,T.tensordot(G.A,w,(1,0)),(0,0))
        else:
            assert(False)
    G.gV = gV
    def cogV(cov=None,cow=None):
        if not cov and not cow:
            return G.W
        elif cov and not cow:
            return T.tensordot(G.W,cov,(1,0))
        elif cov.type == T.vector().type and cow.type == T.vector().type:
            return T.dot(cov,T.dot(G.W,cow))
        elif cov.type == T.matrix().type and cow.type == T.matrix().type:
            return T.tensordot(cov,T.tensordot(G.W,cow,(1,0)),(0,0))
        else:
            assert(False)
    G.cogV = cogV
    def gLA(xiv,xiw):
        v = G.LAtoV(xiv)
        w = G.LAtoV(xiw)
        return G.gV(v,w)
    G.gLA = gLA
    def cogLA(coxiv,coxiw):
        cov = G.LAtoV(coxiv)
        cow = G.LAtoV(coxiw)
        return G.cogV(cov,cow)
    G.cogLA = cogLA
    def gG(g,vg,wg):
        xiv = G.invpb(g,vg)
        xiw = G.invpb(g,wg)
        return G.gLA(xiv,xiw)
    G.gG = gG
    def gpsi(hatxi,v=None,w=None):
        g = G.psi(hatxi)
        vg = G.dpsi(hatxi,v)
        wg = G.dpsi(hatxi,w)
        return G.gG(g,vg,wg)
    G.gpsi = gpsi
    def cogpsi(hatxi,p=None,pp=None):
        invgpsi = G.inv(G.gpsi(hatxi))
        if p and pp:
            return T.tensordot(p,T.tensordot(invgpsi,pp,(1,0)),(0,0))
        elif p and not pp:
            return T.tensordot(invgpsi,p,(1,0))
        return invgpsi
    G.cogpsi = cogpsi
    G.gGf = theano.function([g,vg,wg],G.gG(g,vg,wg))
    G.gpsi_evf = theano.function([hatxi,v,w],G.gpsi(hatxi,v,w))
    G.gpsif = theano.function([hatxi],G.gpsi(hatxi))
    p = G.coordscovector() # \RR^G_dim cotangent vector in coordinates
    pp = G.coordscovector() # \RR^G_dim cotangent vector in coordinates
    G.cogpsi_evf = theano.function([hatxi,p,pp],G.cogpsi(hatxi,p,pp))
    G.cogpsif = theano.function([hatxi],G.cogpsi(hatxi))
    G.gLAf = theano.function([xiv,xiw],G.gLA(xiv,xiw))
    G.gVf = theano.function([v,w],G.gV(v,w))

    # sharp/flat mappings
    def sharpV(mu):
        return T.dot(G.W,mu)
    G.sharpV = sharpV
    def flatV(v):
        return T.dot(G.A,v)
    G.flatV = flatV
    def sharp(g,pg):
        return G.invpf(g,G.VtoLA(T.dot(G.W,G.LAtoV(G.invcopb(g,pg)))))
    G.sharp = sharp
    def flat(g,vg):
        return G.invcopf(g,G.VtoLA(T.dot(G.A,G.LAtoV(G.invpb(g,vg)))))
    G.flat = flat
    def sharppsi(hatxi,p):
        return T.tensordot(G.cogpsi(hatxi),p,(1,0))
    G.sharppsi = sharppsi
    def flatpsi(hatxi,v):
        return T.tensordot(G.gpsi(hatxi),v,(1,0))
    G.flatpsi = flatpsi
    G.sharpVf = theano.function([mu],G.sharpV(mu))
    G.flatVf = theano.function([v],G.flatV(v))
    G.sharpf = theano.function([g,pg],G.sharp(g,pg))
    G.flatf = theano.function([g,vg],G.flat(g,vg))
    G.sharppsif = theano.function([hatxi,p],G.sharppsi(hatxi,p))
    G.flatpsif = theano.function([hatxi,v],G.flatpsi(hatxi,v))

