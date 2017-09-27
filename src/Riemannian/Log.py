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
    """ numerical Riemannian Logarithm map """

    x = M.element()
    y = M.element()
    v = M.vector()

    loss = lambda x,v,y: 1./M.dim.eval()*T.sum(T.sqr(M.Exp(x,v)-y))
    dloss = lambda x,v,y: T.grad(loss(x,v,y),v)
    lossf = theano.function([x,v,y], loss(x,v,y))
    dlossf = theano.function([x,v,y], dloss(x,v,y))

    from scipy.optimize import minimize,fmin_bfgs,fmin_cg
    def shoot(x,y,v0):
        def f(w):
            z = lossf(x,w,y)
            dz = dlossf(x,w,y)
            return (z,dz)

        res = minimize(f, v0, method='L-BFGS-B', jac=True, options={'disp': False, 'maxiter': 100})

        return(res.x,res.fun)

    M.Logf = lambda x,y,v0: shoot(x,y,v0)

