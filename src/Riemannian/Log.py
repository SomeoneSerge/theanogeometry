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

from src.params import *

from src.utils import *

###############################################################
# Logarithm map from provided exponential
###############################################################

# Logarithmic map
loss = lambda q,p,x: 1./d.eval()*T.sum(T.sqr(Exp(q,p)-x))
dloss = lambda q,p,x: T.grad(loss(q,p,x),p)
lossf = theano.function([q,p,x], loss(q,p,x))
dlossf = theano.function([q,p,x], dloss(q,p,x))

from scipy.optimize import minimize,fmin_bfgs,fmin_cg
def shoot(q1,q2,p0):
    def f(x):
        y = lossf(q1,x,q2)
        dy = dlossf(q1,x,q2)
        return (y,dy)
    
    res = minimize(f, p0, method='L-BFGS-B', jac=True, options={'disp': False, 'maxiter': 100})
    
    return(res.x,res.fun)

Logf = lambda q1,q2,p0: shoot(q1,q2,p0)

