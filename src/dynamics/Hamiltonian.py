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

###############################################################
# geodesic integration, Hamiltonian form                      #
###############################################################

def initialize(M):
    q = M.coords()
    p = M.coordscovector()

    dq = lambda q,p: T.grad(M.H(q,p),p)
    dp = lambda q,p: -T.grad(M.H(q,p),q)

    def ode_Hamiltonian(t,x):
        dqt = dq(x[0],x[1])
        dpt = dp(x[0],x[1])
        return T.stack((dqt,dpt))
    M.Hamiltonian_dynamics = lambda q,p: integrate(ode_Hamiltonian,T.stack((q,p)))
    M.Hamiltonian_dynamicsf = theano.function([q,p], M.Hamiltonian_dynamics(q,p))

    ## Geodesic
    M.Exp_Hamiltonian = lambda q,p: M.Hamiltonian_dynamics(q,p)[1][-1,0]
    M.Exp_Hamiltoniant = lambda q,p: M.Hamiltonian_dynamics(q,p)[1][:,0].dimshuffle((1,0))
    M.Exp_Hamiltonianf = theano.function([q,p], M.Exp_Hamiltonian(q,p))
    M.Exp_Hamiltoniantf = theano.function([q,p], M.Exp_Hamiltoniant(q,p))

    ## Group geodesics
    #try:
    #    Exppsi = lambda q,v: Ham(q,flatpsi(q,v))[1][-1,0]
    #    Exptpsi = lambda q,v: Ham(q,flatpsi(q,v))[1][:,0]
    #    DExppsi = lambda q,v: (
    #        T.jacobian(Ham(q,flatpsi(q,v))[1][-1,0].flatten(),q).reshape(N,N,G_dim),
    #        T.jacobian(Ham(q,flatpsi(q,v))[1][-1,0].flatten(),v).reshape(N,N,G_dim)
    #        )
    #    Exp = lambda g,vg: invtrns(g,psi(Ham(zeroV,LAtoV(invpb(g,vg)))[1][-1,0]))
    #    Expt = lambda g,vg: invtrns(g,psi(Ham(zeroV,LAtoV(invpb(g,vg)))[1][:,0].dimshuffle((1,0))))
    #    loss = 1./G_emb_dim*T.sum(T.sqr(Exp(g,vg)-h))
    #    losspsi = 1./G_emb_dim*T.sum(T.sqr(Exppsi(q,v)-h))
    #    dlosspsi = (T.grad(losspsi,q),T.grad(losspsi,v))
    #    Expf = theano.function([g,vg], Exp(g,vg))
    #    Exppsif = theano.function([q,v], Exppsi(q,v))
    #    Exptpsif = theano.function([q,v], Exptpsi(q,v))
    #    #lossf = theano.function([g,vg,h], loss)
    #    #losspsif = theano.function([q,v,h], losspsi)
    #    #dlosspsif = theano.function([q,v,h], [losspsi, dlosspsi[0], dlosspsi[1]])
    #except NameError:
    #    pass

    #
##### Evolution equations:
#dq = lambda q,p: T.grad(H(q,p),p) # Evolution equation for point q in FM.
#dp = lambda q,p: -T.grad(H(q,p),q) # Evolution equation for covector p in FM.
#dqf = theano.function([q,p], dq(q,p))
#dpf = theano.function([q,p], dp(q,p))
#
#def ode_f(qp): # Evolution equations at (p,q).
#    dqt = dq(qp[0],qp[1])
#    dpt = dp(qp[0],qp[1])
#
#    return T.stack((dqt,dpt))
#ode_ff = theano.function([qp], ode_f(qp))
#
#(cout, updates) = theano.scan(fn=integrator(ode_f),
#                              outputs_info=[qp],
#                              n_steps=n_steps)
#
## Compile the Path Evolution:
#simf = function(inputs=[qp],
#                outputs=cout,
#                updates=updates)
#
##### Geodesics on M:
#def Geodesic(q0,p0):
#    gamma_t = simf(np.stack((q0,p0)))
#    return (gamma_t[:,0])
#
#
#def development(gamma0,q0):
#    gamma_t = simfdev(gamma0,q0)
#    return (gamma_t)
#
#

## shooting
#from scipy.optimize import minimize,fmin_bfgs,fmin_cg
#
#def shoot(g,h):
#    def fopts(x):
#        [y,gy] = dlossf(np.stack([q0,x.reshape([N.eval(),G_dim])]).astype(theano.config.floatX))
#        return (y,gy[1].flatten())
#
#    res = minimize(fopts, p0.flatten(), method='L-BFGS-B', jac=True, options={'disp': False, 'maxiter': maxiter})
#
#    return(res.x,res.fun)
#Logf = lambda g,h: shoot(g,h)
#Logpsif = lambda hatm,hata: shoot(psi(hatm),psi(hata))
