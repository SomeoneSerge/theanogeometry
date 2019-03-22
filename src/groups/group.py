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

class LieGroup(EmbeddedManifold):
    """ Base Lie Group class """

    def __init__(self,N=3,invariance='left'):
        EmbeddedManifold.__init__(self)

        self.N = constant(N) # N in SO(N)
        self.emb_dim  = constant(N*N) # matrix/embedding space dimension
        self.invariance = invariance

        self.e = T.eye(N,N) # identity element
        self.zeroLA = T.zeros((N,N)) # zero element in LA

    def initialize(self):
        """ Initial group operations. To be called by sub-classes after definition of dimension, Expm etc. """
        self.zeroV = T.zeros((self.dim,)) # zero element in V
        hatxi = self.Vvector() # \RR^G_dim vector
        xi = self.LAvector() # matrix in LA
        eta = self.LAvector() # matrix in LA
        alpha = self.LAcovector() # matrix in LA^*
        beta = self.LAcovector() # matrix in LA^*
        g = self.element() # \RR^{NxN} matrix
        gs = self.elements() # sequence of \RR^{NxN} matrices
        h = self.element() # \RR^{NxN} matrix
        vg = self.vector() # \RR^{NxN} tangent vector at g
        wg = self.vector() # \RR^{NxN} tangent vector at g
        vh = self.vector() # \RR^{NxN} tangent vector at h
        w = self.coordsvector() # \RR^G_dim tangent vector in coordinates
        v = self.coordsvector() # \RR^G_dim tangent vector in coordinates
        pg = self.covector() # \RR^{NxN} cotangent vector at g
        ph = self.covector() # \RR^{NxN} cotangent vector at h
        p = self.coordscovector() # \RR^G_dim cotangent vector in coordinates
        pp = self.coordscovector() # \RR^G_dim cotangent vector in coordinates
        mu = self.Vcovector() # \RR^G_dim LA cotangent vector in coordinates

        # compile group specific chart functions
        self.VtoLAf = theano.function([hatxi], self.VtoLA(hatxi))
        self.LAtoVf = theano.function([xi], self.LAtoV(xi))

        ## group operations
        self.inv = lambda g: T.nlinalg.MatrixInverse()(g)
        self.invf = theano.function([g],self.inv(g))

        ## group exp/log maps
        self.exp = self.Expm
        self.expf = theano.function([xi],self.exp(xi))
        def expt(xi):
            (cout, updates) = theano.scan(fn=lambda t,x,dt,xi: (t+dt,self.exp(t*xi)),
                                          outputs_info=[constant(0.),self.e],
                                          non_sequences=[dt,xi],
                                          n_steps=n_steps)
            return cout
        self.expt = expt
        self.exptf = theano.function([xi],self.expt(xi))
        self.log = self.Logm
        self.logf = theano.function([g],self.log(g))

        ## shooting
        #from scipy.optimize import minimize,fmin_bfgs,fmin_cg
        #logm_loss = lambda xi: (1./2.)*T.sum(T.sqr(exp(xi)-h))
        #dlogm_loss = lambda xi: T.grad((1./2.)*T.sum(T.sqr(exp(xi)-h)),xi)
        #logm_lossf = theano.function([xi],logm_loss(xi))
        #dlogm_lossf = theano.function([xi],(logm_loss(xi),dlogm_loss(xi)))
        #def shoot_logm(g,h):
        #    res = minimize(dlogm_lossf, np.zeros((G_dim,G_dim)), method='L-BFGS-B', jac=True, options={'disp': False, 'maxiter': maxiter})
        #
        #    return(res.x,res.fun)
        #logmf = lambda g,h: shoot_logm(g,h)
        #logpsif = lambda xi,eta: shoot(psi(xi),psi(eta))

        ## Lie algebra
        self.eiV = T.eye(self.dim) # standard basis for V
        self.eiLA = self.VtoLA(self.eiV) # pushforward eiV basis for LA
        #stdLA = T.eye(N*N,N*N).reshape((N,N,N*N)) # standard basis for \RR^{NxN}
        #eijV = theano.shared(np.eye(G_dim.eval())) # standard basis for V
        #eijLA = theano.shared(np.zeros((int(N.eval()),int(N.eval()),int(G_dim.eval())))) # eij in LA
        def bracket(xi,eta):
            if xi.type == T.matrix().type and eta.type == T.matrix().type:
                return T.tensordot(xi,eta,(1,0))-T.tensordot(eta,xi,(1,0))
            elif xi.type == T.tensor3().type and eta.type == T.tensor3().type:
                return T.tensordot(xi,eta,(1,0)).dimshuffle((0,2,1,3))-T.tensordot(eta,xi,(1,0)).dimshuffle((0,2,1,3))
            else:
                assert(False)
        self.bracket =  bracket
        self.bracketf = theano.function([xi,eta],self.bracket(xi,eta))
        #C = bracket(eiLA,eiLA) # structure constants, debug
        #C = T.nlinalg.lstsq(eiLA.reshape((N*N*G_dim*G_dim,G_dim*G_dim*G_dim)),bracket(eiLA,eiLA).reshape((N*N*G_dim*G_dim))).reshape((G_dim,G_dim,G_dim)) # structure constants
        #Cf = theano.function([],C)
        self.C = theano.shared(np.zeros((self.dim.eval(),self.dim.eval(),self.dim.eval()))) # structure constants
        lC = np.zeros((self.dim.eval(),self.dim.eval(),self.dim.eval()))
        for i in range(self.dim.eval()):
            for j in range(self.dim.eval()):
                xij = self.bracket(self.eiLA[:,:,i],self.eiLA[:,:,j])
                #lC[i,j,:] = T.nlinalg.lstsq()(
                #    self.eiLA.reshape((self.N*self.N,self.dim)),
                #    xij.flatten(),
                #    rcond=-1
                #)[0].eval()
                lC[i,j,:] = np.linalg.lstsq(
                        self.eiLA.eval().reshape(self.N.eval()*self.N.eval(), self.dim.eval()),
                        xij.eval().reshape(self.N.eval()*self.N.eval())
                        )[0]
        self.C.set_value(lC)

        ## surjective mapping \psi:\RR^G_dim\rightarrow G
        self.psi = lambda hatxi: self.exp(self.VtoLA(hatxi))
        self.invpsi = lambda g: self.LAtoV(self.log(g))
        def dpsi(hatxi,v):
            dpsi = T.jacobian(self.exp(self.VtoLA(hatxi)).flatten(),hatxi).reshape((self.N,self.N,self.dim))
            if v:
                return T.tensordot(dpsi,v,(2,0))
            return dpsi
        self.dpsi = dpsi
        self.psif = theano.function([hatxi],self.psi(hatxi))
        self.invpsif = theano.function([g],self.invpsi(g))
        self.dpsif = theano.function([hatxi,v],self.dpsi(hatxi,v))

        ## left/right translation
        self.L = lambda g,h: T.tensordot(g,h,(1,0)) # left translation L_g(h)=gh
        self.R = lambda g,h: T.tensordot(h,g,(1,0)) # right translation R_g(h)=hg
        # pushforward of L/R of vh\in T_hG
        #dL = lambda g,h,vh: theano.gradient.Rop(L(theano.gradient.disconnected_grad(g),h).flatten(),h,vh).reshape((N,N))
        def dL(g,h,vh=None):
            dL = T.jacobian(self.L(theano.gradient.disconnected_grad(g),h).flatten(),h).reshape((self.N,self.N,self.N,self.N))
            if vh:
                return T.tensordot(dL,vh,((2,3),(0,1)))
            return dL
        self.dL = dL
        #dR = lambda g,h,vh: theano.gradient.Rop(R(theano.gradient.disconnected_grad(g),h).flatten(),h,vh).reshape((N,N))
        self.dR = lambda g,h,vh: T.tensordot(T.jacobian(self.R(theano.gradient.disconnected_grad(g),h).flatten(),h),vh,((1,2),(0,1))).reshape((self.N,self.N))
        def dR(g,h,vh=None):
            dR = T.jacobian(self.R(theano.gradient.disconnected_grad(g),h).flatten(),h).reshape((self.N,self.N,self.N,self.N))
            if vh:
                return T.tensordot(dR,vh,((2,3),(0,1)))
            return dR
        self.dR = dR
        # pullback of L/R of vh\in T_h^*G
        self.codL = lambda g,h,vh: self.dL(g,h,vh).T
        self.codR = lambda g,h,vh: self.dR(g,h,vh).T
        self.dLf = theano.function([g,h,vh],self.dL(g,h,vh))
        self.dRf = theano.function([g,h,vh],self.dR(g,h,vh))
        self.codLf = theano.function([g,h,ph],self.codL(g,h,ph))
        self.codRf = theano.function([g,h,ph],self.codR(g,h,ph))

        ## actions
        self.Ad = lambda g,xi: self.dR(self.inv(g),g,self.dL(g,self.e,xi))
        self.ad = lambda xi,eta: self.bracket(xi,eta)
        self.coad = lambda p,pp: T.tensordot(T.tensordot(self.C,p,(0,0)),pp,(1,0)) # TODO: check this
        self.Adf = theano.function([g,xi],self.Ad(g,xi))
        self.adf = theano.function([xi,eta],self.ad(xi,eta))
        self.coadf = theano.function([p,pp],self.coad(p,pp))

        ## invariance
        if self.invariance == 'left':
            self.invtrns = self.L # invariance translation
            self.invpb = lambda g,vg: self.dL(self.inv(g),g,vg) # left invariance pullback from TgG to LA
            self.invpf = lambda g,xi: self.dL(g,self.e,xi) # left invariance pushforward from LA to TgG
            self.invcopb = lambda g,pg: self.codL(self.inv(g),g,pg) # left invariance pullback from Tg^*G to LA^*
            self.invcopf = lambda g,alpha: self.codL(g,self.e,alpha) # left invariance pushforward from LA^* to Tg^*G
            self.infgen = lambda xi,g: self.dR(g,self.e,xi) # infinitesimal generator
        else:
            self.invtrns = self.R # invariance translation
            self.invpb = lambda g,vg: self.dR(self.inv(g),g,vg) # right invariance pullback from TgG to LA
            self.invpf = lambda g,xi: self.dR(g,self.e,xi) # right invariance pushforward from LA to TgG
            self.invcopb = lambda g,pg: self.codR(self.inv(g),g,pg) # right invariance pullback from Tg^*G to LA^*
            self.invcopf = lambda g,alpha: self.codR(g,self.e,alpha) # right invariance pushforward from LA^* to Tg^*G
            self.infgen = lambda xi,g: self.dL(g,self.e,xi) # infinitesimal generator
        self.invpbf = theano.function([g,vg],self.invpb(g,vg))
        self.invpff = theano.function([g,xi],self.invpf(g,xi))
        self.invcopbf = theano.function([g,pg],self.invcopb(g,pg))
        self.invcopff = theano.function([g,alpha],self.invcopf(g,alpha))
        self.infgenf = theano.function([xi,g],self.infgen(xi,g))

    def element(self):
        """ return symbolic element in manifold """
        return T.matrix()

    def elements(self):
        """ return symbolic sequence of elements in manifold """
        return T.tensor3()

    def coords(self):
        """ return symbolic coordinate representation of point in manifold """
        return T.vector()

    def vector(self):
        """ return symbolic tangent vector """
        return T.matrix()

    def vectors(self):
        """ return symbolic sequence of tangent vector """
        return T.tensor3()

    def covector(self):
        """ return symbolic cotangent vector """
        return T.matrix()

    def coordsvector(self):
        """ return symbolic tangent vector in coordinate representation """
        return T.vector()

    def coordscovector(self):
        """ return symbolic cotangent vector in coordinate representation """
        return T.vector()

    def frame(self):
        """ return symbolic frame for tangent space """
        return T.tensor3()

    def LAvector(self):
        """ return symbolic Lie Algebra vector """
        return T.matrix()

    def LAcovector(self):
        """ return symbolic Lie Algebra covector """
        return T.matrix()

    def Vvector(self):
        """ return symbolic vector in V vector space (basis representation of Lie Algebra) """
        return T.vector()

    def Vcovector(self):
        """ return symbolic covector in V vector space (basis representation of Lie Algebra) """
        return T.vector()

    def __str__(self):
        return "abstract Lie group"

