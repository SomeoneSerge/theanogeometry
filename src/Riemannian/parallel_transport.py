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
    """ Riemannian parallel transport """

    d = M.dim
    x = M.element()
    u = M.frame()

    def ode_parallel_transport(gamma,dgamma,t,x):
        dpt = - T.tensordot(T.tensordot(dgamma, M.Gamma_g(gamma),axes = [0,1]),
                            x, axes = [1,0])
        return dpt

    parallel_transport = lambda v,gamma,dgamma: integrate(ode_parallel_transport,v,gamma,dgamma)
    M.parallel_transport = lambda v,gamma,dgamma: parallel_transport(v,gamma,dgamma)[1]
    v = M.vector()
    gamma = M.elements()
    dgamma = M.vectors()
    M.parallel_transportf = theano.function([v,gamma,dgamma], M.parallel_transport(v,gamma,dgamma))

