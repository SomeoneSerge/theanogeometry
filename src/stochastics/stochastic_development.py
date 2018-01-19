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
    """ development and stochastic development from R^d to M """

    t = T.scalar()
    dgamma = T.matrix() # velocity of Euclidean curve
    dsm = T.matrix() # derivative of Euclidean semimartingale
    u = M.FM_element()
    d = M.dim

    # Deterministic development
    def ode_development(dgamma,t,u):
        x = u[0:d]
        nu = u[d:].reshape((d,-1))
        m = nu.shape[1]

        det = T.tensordot(M.Horizontal(u)[:,0:m], dgamma, axes = [1,0])
    
        return det

    M.development = lambda u,dgamma: integrate(ode_development,u,dgamma)
    M.developmentf = theano.function([u,dgamma], M.development(u,dgamma))

    # Stochastic development
    def sde_development(dsm,t,u):
        x = u[0:d]
        nu = u[d:].reshape((d,-1))
        m = nu.shape[1]

        sto = T.tensordot(M.Horizontal(u)[:,0:m], dsm, axes = [1,0])
    
        return (T.zeros_like(sto), sto, M.Horizontal(u)[:,0:m])

    M.sde_development = sde_development
    M.sde_developmentf = theano.function([dsm,t,u], M.sde_development(dsm,t,u), on_unused_input = 'ignore') 
    M.stochastic_development = lambda u,dsm: integrate_sde(sde_development,integrator_stratonovich,u,dsm)
    M.stochastic_developmentf = theano.function([u,dsm], M.stochastic_development(u,dsm))

