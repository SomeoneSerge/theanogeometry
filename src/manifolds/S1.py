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

from src.manifolds.manifold import *

#from src.plotting import *
import matplotlib.pyplot as plt

class S1(EmbeddedManifold):
    """ 1d circle """

    def __init__(self):
        
        def F(x): 
            #u = x[0]/(1.0-x[1])
            return T.stack([T.sin(x[0]),T.cos(x[0])])#2*u/(1.+u**2),(1-u**2)/(1+u**2)])

        EmbeddedManifold.__init__(self,F,1,2) # 2,3?

        # hardcoded Jacobian for speed (removing one layer of differentiation)
        def JF(self,x):
            #u = x[0]/(1.0-x[1])
            return T.stack([T.cos(x),-T.sin(x)])#(2-3*u**2)/(1+u**2)**2,(2-3*u**2)/(1+u**2)**2*(u/(1-x[1])),-4*u/(1.0+u**2)**2,-4*u**2/(1+u**2)**2*(1./(1-x[1]))]).reshape((2,2))

        x = self.coords()
        self.JF_theanof = self.JFf
        self.JFf = theano.function([x], self.JF(x))
        # metric matrix
        self.g = lambda x: T.dot(self.JF(x).T,self.JF(x))

    def __str__(self):
        return "%dd circle" % (self.dim.eval())

    def plot(self,alpha=None,lw=0.3):
        
        t = np.arange(0,2*np.pi, 1./100)
        plt.plot(np.cos(t),np.sin(t), linewidth = 1, color = 'gray', zorder=1)

        plt.axis('equal')
    
 # plot x on ellipsoid. x can be either in coordinates or in R^3
    def plotx(self, x, i0=0, color='b', color_intensity=1., linewidth=1., s=15., prevx=None, last=True):
        if len(x.shape)>1:
            for i in range(x.shape[0]):
                self.plotx(x[i],i0=i,
                           color=color,
                           color_intensity=color_intensity if i==0 or i==x.shape[0]-1 else .7,
                           linewidth=linewidth,
                           s=s,
                           prevx=x[i-1] if i>0 else None,
                           last=i==(x.shape[0]-1))
            return

        xcoords = x
        x = self.Ff(x)

        if prevx is None or last:
            plt.scatter(x[0],x[1],color=color,s=s, zorder=2)
        if prevx is not None:
            prevx = self.Ff(prevx)
            xx = np.stack((prevx,x))
            plt.plot(xx[:,0],xx[:,1],linewidth=linewidth,color=color)

