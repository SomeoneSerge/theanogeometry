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
#from src.manifold import *

def initialize(M):

    #from src.framebundle import FM
    #FM.initialize(M)
    
    d = M.dim
    m = M.rank
    #n = M.N

    dgamma = M.process() # deterministic curve
    dW = M.process() # stochastic process
    drift = M.vector()
    q = M.element()

    ## Development: (Deterministic)
    def ode_Dev(dgamma,t,q):
    
        dgamma1 = dgamma #T.tile(T.repeat(dgamma.reshape((M.m,M.rank//M.m)),M.m,axis=1),(M.m*6)//M.rank)

        x = q[0:d]
        ui = q[d:(d+d*d)].reshape((d,d)) #q[d:(d+n*d)].reshape((d,d))
        m = dgamma.shape[0]

        det = T.tensordot(M.Hori(x,ui)[:,0:m], dgamma1, axes = [1,0])
    
        return det

    M.dev = lambda q,dgamma: integrate(ode_Dev,q,dgamma)[1]
    M.devf = theano.function([q,dgamma], M.dev(q,dgamma))

    # Stochastic Development:
    def sde_SD(dWt,t,q,drift):
        
        dWt1 = dWt#T.tile(T.repeat(dWt.reshape((M.m,M.rank//M.m)),M.m,axis=1),(M.m*6)//M.rank)
        drift1 = drift #T.tile(T.repeat(drift.reshape((M.m,M.rank//M.m)),M.m,axis=1),(M.m*6)//M.rank)
        
        x = q[0:d]
        ui = q[d:(d+d*d)].reshape((d,d)) #q[d:(d+n*d)].reshape((d,n))
        m = dWt1.shape[0]

        det = T.tensordot(M.Hori(x,ui)[:,0:m], drift1, axes = [1,0]) #T.diagonal(T.tensordot(M.Hori(x,ui), drift1, axes = [1,0]))
        sto = T.tensordot(M.Hori(x,ui)[:,0:m], dWt1, axes = [1,0]) #T.diagonal(T.tensordot(M.Hori(x,ui), dWt1, axes = [1,0]))
    
        return (det, sto, T.constant(0.), T.constant(0.))

    M.stoc_dev = lambda q,dWt,drift: integrate_sde(sde_SD,integrator_stratonovich,q,dWt,drift)[1]
    M.stoc_devf = theano.function([q,dWt,drift], M.stoc_dev(q,dWt,drift))

