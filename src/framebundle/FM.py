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
#from src.manifold import *
#from src.metric import *
from src.utils import *

def initialize(M):
    """ Frame bundle geometry """ 
    
    d  = M.dim
    r = M.rank # dimension used for development

    x = M.element()
    x1 = M.element()
    v = M.vector()
    u = M.frame()
    q = M.element()
    p = M.covector()

    ##### Cometric matrix:
    def gFMsharp(u):
    
        x = u[0:d]
        ui = u[d:].reshape((d,M.m))
        GamX = T.tensordot(M.Gamma_g(x), ui, 
                           axes = [2,0]).dimshuffle(0,2,1)
    
        delta = T.eye(ui.shape[0],ui.shape[1])
        W = T.tensordot(ui, ui, axes = [1,1]) + lambdag0*M.g(x)
    
        gij = W
        gijb = -T.tensordot(W, GamX, axes = [1,2])
        giaj = -T.tensordot(GamX, W, axes = [2,0])
        giajb = T.tensordot(T.tensordot(GamX, W, axes = [2,0]), 
                            GamX, axes = [2,2])

        return gij,gijb,giaj,giajb

    ##### Hamiltonian on FM based on the pseudo metric tensor: 
    lambdag0 = 0

    xi = T.vector()
    xia = T.matrix()
    def Hsplit(x,ui,xi,xia):
        
        GamX = T.tensordot(M.Gamma_g(x), ui, 
                           axes = [2,0]).dimshuffle(0,2,1)
    
        delta = T.eye(ui.shape[0],ui.shape[1])
        W = T.tensordot(ui, ui, axes = [1,1]) + lambdag0*M.g(x)
    
        gij = W
        gijb = -T.tensordot(W, GamX, axes = [1,2])
        giaj = -T.tensordot(GamX, W, axes = [2,0])
        giajb = T.tensordot(T.tensordot(GamX, W, axes = [2,0]), 
                            GamX, axes = [2,2])
    
        xigxi = T.dot(T.tensordot(xi, gij, axes = [0,0]), xi)
        xigxia = T.tensordot(T.tensordot(xi, gijb, axes = [0,0]), 
                             xia, axes = [[0,1],[0,1]])
        xiagxi = T.tensordot(T.tensordot(xi, giaj, axes = [0,2]), 
                             xia, axes = [[0,1],[0,1]])
        xiagxia = T.tensordot(T.tensordot(giajb, xia, axes = [[2,3],[0,1]]), 
                              xia, axes = [[0,1],[0,1]])
    
        return 0.5*(xigxi + xigxia + xiagxi + xiagxia)

    M.Hfm = lambda q,p: Hsplit(q[0:d],q[d:(d+M.m*d)].reshape((d,M.m)),\
                               p[0:d],p[d:(d+M.m*d)].reshape((d,M.m)))
    M.Hfmf = theano.function([q,p],M.Hfm(q,p))

    ##### Evolution equations:
    dq = lambda q,p: T.grad(M.Hfm(q,p),p)
    dp = lambda q,p: -T.grad(M.Hfm(q,p),q)
    #dqfmf = theano.function([q,p], dq(q,p))
    #dpfmf = theano.function([q,p], dp(q,p))

    def ode_Hamfm(t,x): # Evolution equations at (p,q).
        dqt = dq(x[0],x[1])
        dpt = dp(x[0],x[1])
        return T.stack((dqt,dpt))
    M.Hamfm = lambda q,p: integrate(ode_Hamfm,T.stack((q,p)))
    M.Hamfmf = theano.function([q,p], M.Hamfm(q,p))

    ## Geodesic
    M.Expfm = lambda q,p: M.Hamfm(q,p)[1][-1,0]
    M.Exptfm = lambda q,p: M.Hamfm(q,p)[1][:,0].dimshuffle((1,0))
    M.Expfmf = theano.function([q,p], M.Expfm(q,p))
    M.Exptfmf = theano.function([q,p], M.Exptfm(q,p))

    # Most Probable Path
    M.loss = lambda x,x1,v: 1./d*T.sum((M.Expfm(x,M.flat(x,v))[0:d] - x1[0:d])**2)
    M.lossf = theano.function([x,x1,v], M.loss(x,x1,v))

    def Logfm(x,x1):
        def fopts(v):
            y = M.lossf(x,x1,v)
            return y

        res = minimize(fopts, np.zeros([d.eval()+M.m.eval()*d.eval()]), 
                       method='CG', jac=False, options={'disp': False, 
                                                        'maxiter': 50})
        return res.x

    ##### Horizontal vector fields:
    def Hori(x,u):
    
        # Contribution from the coordinate basis for x: 
        dx = u
        # Contribution from the basis for Xa:
        du = -T.tensordot(u, T.tensordot(u, M.Gamma_g(x), axes = [0,2]), axes = [0,2])

        duv = du.reshape((u.shape[1],du.shape[1]*du.shape[2]))

        return T.concatenate([dx,duv.T], axis = 0)
    M.Hori = Hori
    M.Horif = theano.function([x,u],M.Hori(x,u))

