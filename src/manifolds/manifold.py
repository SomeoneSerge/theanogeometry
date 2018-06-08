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

class Manifold(object):
    """ Base manifold class """

    def __init__(self):
        self.dim = None

    def __init__(self):
        self.rank = None

    def element(self):
        """ return symbolic element in manifold """
        return T.vector()

    def elements(self):
        """ return symbolic sequence of elements in manifold """
        return T.matrix()

    def coords(self):
        """ return symbolic coordinate representation of point in manifold """
        return T.vector()

    def vector(self):
        """ return symbolic tangent vector """
        return T.vector()

    def vectors(self):
        """ return symbolic sequence of tangent vector """
        return T.matrix()

    def covector(self):
        """ return symbolic cotangent vector """
        return T.vector()

    def coordsvector(self):
        """ return symbolic tangent vector in coordinate representation """
        return T.vector()

    def coordscovector(self):
        """ return symbolic cotangent vector in coordinate representation """
        return T.vector()

    def frame(self):
        """ return symbolic frame for tangent space """
        return T.matrix()

    def process(self):
        """ return symbolic steps of process """
        return T.matrix()

    def newfig(self):
        """ open new plot for manifold """

    def __str__(self):
        return "abstract manifold"

class EmbeddedManifold(Manifold):
    """ Embedded manifold base class """

    def __init__(self,F,dim,emb_dim,invF=None):
        Manifold.__init__(self)
        self.F = F
        self.invF = invF
        self.dim = constant(dim)
        self.emb_dim = constant(emb_dim)

        x = self.coords()
        self.Ff = theano.function([x], self.F(x))

        self.JF = lambda x: T.jacobian(self.F(x),x)
        self.JFf = theano.function([x], self.JF(x))

        # metric matrix
        self.g = lambda x: T.dot(self.JF(x).T,self.JF(x))


        # get coordinate representation from embedding space
        from scipy.optimize import minimize
        def get_get_coords():
            x = self.element()
            y = self.element()
        
            loss = lambda x,y: 1./self.emb_dim.eval()*T.sum(T.sqr(self.F(x)-y))
            dloss = lambda x,y: T.grad(loss(x,y),x)
            dlossf = theano.function([x,y], (loss(x,y),dloss(x,y)))
        
            from scipy.optimize import minimize,fmin_bfgs,fmin_cg
            def get_coords(y,x0=None):        
                def f(x):
                    (z,dz) = dlossf(x,y.astype(theano.config.floatX))
                    return (z.astype(np.double),dz.astype(np.double))
                if x0 is None:
                    x0 = np.zeros(self.dim.eval()).astype(np.double)
                res = minimize(f, x0, method='CG', jac=True, options={'disp': False, 'maxiter': 100})
                return res.x
            
            return get_coords
        self.get_coordsf = get_get_coords()

    def __str__(self):
        return "dim %d manifold embedded in R^%d" % (self.dim.eval(),self.emb_dim.eval())
