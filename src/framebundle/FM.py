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
    """ Frame Bundle geometry """
    
    d  = M.dim

    x = M.element()
    x1 = M.element()
    v = M.vector()
    nu = M.frame()

    def FM_element():
        """ return element of FM as concatenation (x,nu) flattened """
        return T.vector()
    def FM_vector():
        """ vector in TFM """
        return T.vector()
    def FM_covector():
        """ covector in T^*FM """
        return T.vector()
    M.FM_element = FM_element
    M.FM_vector = FM_vector
    M.FM_covector = FM_covector

    u = M.FM_element()
    q = M.FM_element()
    p = M.FM_covector()

    ##### Cometric matrix:
    def g_FMsharp(u):
        x = u[0:d]
        nu = u[d:].reshape((d,d))#.reshape((d,M.m))
        GamX = T.tensordot(M.Gamma_g(x), nu, axes = [2,0]).dimshuffle(0,2,1)
    
        delta = T.eye(nu.shape[0],nu.shape[1])
        W = T.tensordot(nu,  nu,  axes = [1,1]) + lambdag0*M.g(x)
    
        gij = W
        gijb = -T.tensordot(W, GamX, axes = [1,2])
        giaj = -T.tensordot(GamX, W, axes = [2,0])
        giajb = T.tensordot(T.tensordot(GamX, W, axes = [2,0]), 
                            GamX, axes = [2,2])

        return gij,gijb,giaj,giajb

    ##### Hamiltonian on FM based on the pseudo metric tensor: 
    lambdag0 = 0

    def H_FM(x,nu,px,pnu):
        
        GamX = T.tensordot(M.Gamma_g(x), nu, 
                           axes = [2,0]).dimshuffle(0,2,1)
    
        delta = T.eye(nu.shape[0],nu.shape[1])
        W = T.tensordot(nu, nu, axes = [1,1]) + lambdag0*M.g(x)
    
        gij = W
        gijb = -T.tensordot(W, GamX, axes = [1,2])
        giaj = -T.tensordot(GamX, W, axes = [2,0])
        giajb = T.tensordot(T.tensordot(GamX, W, axes = [2,0]), 
                            GamX, axes = [2,2])
    
        pxgpx = T.dot(T.tensordot(px, gij, axes = [0,0]), px)
        pxgpnu = T.tensordot(T.tensordot(px, gijb, axes = [0,0]), 
                             pnu, axes = [[0,1],[0,1]])
        pnugpx = T.tensordot(T.tensordot(px, giaj, axes = [0,2]), 
                             pnu, axes = [[0,1],[0,1]])
        pnugpnu = T.tensordot(T.tensordot(giajb, pnu, axes = [[2,3],[0,1]]), 
                              pnu, axes = [[0,1],[0,1]])
    
        return 0.5*(pxgpx + pxgpnu + pnugpx + pnugpnu)

    M.H_FM = lambda q,p: H_FM(q[0:d],q[d:].reshape((d,-1)),\
                               p[0:d],p[d:].reshape((d,-1)))
    M.H_FMf = theano.function([q,p],M.H_FM(q,p))

    ##### Evolution equations:
    dq = lambda q,p: T.grad(M.H_FM(q,p),p)
    dp = lambda q,p: -T.grad(M.H_FM(q,p),q)

    def ode_Hamiltonian_FM(t,x): # Evolution equations at (p,q).
        dqt = dq(x[0],x[1])
        dpt = dp(x[0],x[1])
        return T.stack((dqt,dpt))
    M.Hamiltonian_dynamics_FM = lambda q,p: integrate(ode_Hamiltonian_FM,T.stack((q,p)))
    M.Hamiltonian_dynamics_FMf = theano.function([q,p], M.Hamiltonian_dynamics_FM(q,p))

    ## Geodesic
    M.Exp_Hamiltonian_FM = lambda q,p: M.Hamiltonian_dynamics_FM(q,p)[1][-1,0]
    M.Exp_Hamiltonian_FMt = lambda q,p: M.Hamiltonian_dynamics_FM(q,p)[1][:,0].dimshuffle((1,0))
    M.Exp_Hamiltonian_FMf = theano.function([q,p], M.Exp_Hamiltonian_FM(q,p))
    M.Exp_Hamiltonian_FMtf = theano.function([q,p], M.Exp_Hamiltonian_FMt(q,p))

    # Most probable path for the driving semi-martingale
    M.loss = lambda u,x,p: 1./d*T.sum((M.Exp_Hamiltonian_FM(u,p)[0:d] - x[0:d])**2)
    M.lossf = theano.function([u,x,p], M.loss(u,x,p))

    def Log_FM(u,x):
        def fopts(p):
            y = M.lossf(u,x,p)
            return y

        res = minimize(fopts, np.zeros(u.shape), 
                       method='CG', jac=False, options={'disp': False, 
                                                        'maxiter': 50})
        return res.x
    M.Log_FM = Log_FM

    ##### Horizontal vector fields:
    def Horizontal(u):
        x = u[0:d]
        nu = u[d:].reshape((d,-1))
    
        # Contribution from the coordinate basis for x: 
        dx = nu
        # Contribution from the basis for Xa:
        dnu = -T.tensordot(nu, T.tensordot(nu, M.Gamma_g(x), axes = [0,2]), axes = [0,2])

        dnuv = dnu.reshape((nu.shape[1],dnu.shape[1]*dnu.shape[2]))

        return T.concatenate([dx,dnuv.T], axis = 0)
    M.Horizontal = Horizontal
    M.Horizontalf = theano.function([u],M.Horizontal(u))

