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

    d = M.dim

    dgamma = M.process() # deterministic curve
    dW = M.process() # stochastic process
    drift = M.vector()
    u = M.element()

    ## Development: (Deterministic)
    def ode_Dev(dgamma,t,u):
    
        x = u[0:d]
        ui = u[d:(d+d*d)].reshape((d,d)) 
        m = dgamma.shape[0]

        det = T.tensordot(M.Hori(x,ui)[:,0:m], dgamma, axes = [1,0])
    
        return det

    M.dev = lambda u,dgamma: integrate(ode_Dev,u,dgamma)[1]
    M.devf = theano.function([u,dgamma], M.dev(u,dgamma))

    # Stochastic Development:
    def sde_SD(dWt,t,u,drift):
        
        x = u[0:d]
        ui = u[d:(d+d*d)].reshape((d,d))
        m = dWt.shape[0]

        det = T.tensordot(M.Hori(x,ui)[:,0:m], drift, axes = [1,0])
        sto = T.tensordot(M.Hori(x,ui)[:,0:m], dWt, axes = [1,0])
    
        return (det, sto, T.constant(0.), T.constant(0.))

    M.stoc_dev = lambda u,dWt,drift: integrate_sde(sde_SD,integrator_stratonovich,u,dWt,drift)[1]
    M.stoc_devf = theano.function([u,dWt,drift], M.stoc_dev(u,dWt,drift))

