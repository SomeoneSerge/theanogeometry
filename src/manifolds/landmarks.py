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

import matplotlib.pyplot as plt

class landmarks(Manifold):
    """ LDDMM landmark manifold """

    def __init__(self,N=1,m=2,k_alpha=1.,k_sigma=np.diag((.5,.5))):
        Manifold.__init__(self)

        self.N = theano.shared(N) # number of landmarks
        self.m = constant(m) # landmark space dimension (usually 2 or 3
        self.dim = self.m*self.N
        self.rank = theano.shared(self.dim.eval())

        self.k_alpha = theano.shared(k_alpha)
        self.k_sigma = theano.shared(k_sigma) # standard deviation of the kernel
        self.inv_k_sigma = theano.tensor.nlinalg.MatrixInverse()(self.k_sigma)
        self.k_Sigma = T.tensordot(k_sigma,k_sigma,(1,1))

        ##### Kernel on M:
        def k(q1,q2):
            r_sq = T.sqr(T.tensordot(q1.reshape((-1,m)).dimshuffle(0,'x',1)-q2.reshape((-1,m)).dimshuffle('x',0,1),self.inv_k_sigma,(2,1))).sum(2)
            return  self.k_alpha*T.exp(-.5*r_sq)
        self.k = k

        # in coordinates
        self.K = lambda q1,q2: (self.k(q1,q2)[:,:,np.newaxis,np.newaxis]*T.eye(self.m)[np.newaxis,np.newaxis,:,:]).dimshuffle((0,2,1,3)).reshape((-1,self.dim))
        q1 = self.element()
        q2 = self.element()
        Kf = theano.function([q1,q2],self.K(q1,q2))

        ##### Metric:
        def gsharp(q):
            return self.K(q,q)
        self.gsharp = gsharp

    def __str__(self):
        return "%d landmarks in R^%d (dim %d). k_alpha=%d, k_sigma=%s" % (self.N.eval(),self.m.eval(),self.dim.eval(),self.k_alpha.eval(),self.k_sigma.eval())

    def plot(self):
        plt.axis('equal')

    def plotx(self, x, u=None, color='b', color_intensity=1., linewidth=1., prevx=None, last=True, curve=False, markersize=None):
        if len(x.shape)>1:
            for i in range(x.shape[0]):
                self.plotx(x[i], u=u if i == 0 else None,
                           color=color,
                           color_intensity=color_intensity if i==0 or i==x.shape[0]-1 else .7,
                           prevx=x[i-1] if i>0 else None,
                           last=i==(x.shape[0]-1),
                           curve=curve)
            return

        x = x.reshape((-1,self.m.eval()))
        NN = x.shape[0]

        for j in range(NN):
            if prevx is None or last:
                plt.scatter(x[j,0],x[j,1],color=color,s=markersize)
            if prevx is not None:
                prevx = prevx.reshape((NN,self.m.eval()))
                xx = np.stack((prevx[j,:],x[j,:]))
                plt.plot(xx[:,0],xx[:,1],linewidth=linewidth,color=color)

            if u is not None:
                u = u.reshape((NN, self.m.eval()))
                plt.quiver(x[j,0], x[j,1], u[j, 0], u[j, 1], pivot='tail', linewidth=linewidth, scale=5)
        if curve and (last or prevx is None):
            plt.plot(np.hstack((x[:,0],x[0,0])),np.hstack((x[:,1],x[0,1])),'o-',color=color)

    # plot point in frame bundle FM
    def plotFMx(self,q,N_vec=None,i0=0,color='b',s=10,color_intensity=1.,linewidth=3.,prevx=None,last=True):

        if len(q.shape)>1:
            for i in range(q.shape[0]):
                self.plotFMx(q[i],
                        N_vec=N_vec,i0=i,
                        color=color, s=s,
                        linewidth=linewidth if i==0 or i==q.shape[0]-1 else .8,
                        color_intensity=color_intensity if i==0 or i==q.shape[0]-1 else .7,
                        prevx=q[i-1] if i>0 else None,
                        last=i==(q.shape[0]-1))
            return

        x = q[0:self.dim.eval()].reshape((self.N.eval(),2))
        ui = q[self.dim.eval():].reshape((self.m.eval(),self.N.eval(),self.m.eval()))

        for j in range(self.N.eval()):
            if prevx is None or last:
                plt.scatter(x[j,0],x[j,1],color=color)
            if prevx is not None:
                prevxx = prevx[0:self.dim.eval()].reshape((self.N.eval(),2))
                xx = np.stack((prevxx[j,:],x[j,:]))
                plt.plot(xx[:,0],xx[:,1],linewidth=linewidth,color=color)

            if N_vec is not None:
                Seq = lambda m, n: [t*n//m + n//(2*m) for t in range(m)]
                Seqv = np.hstack((0,Seq(N_vec,n_steps.eval())))
                if i0 in Seqv:
                    plt.quiver(x[j,0],x[j,1],ui[j,0,0],
                               ui[j,0,1],pivot='tail',linewidth=linewidth,scale=s)
                    plt.quiver(x[j,0],x[j,1],ui[j,1,0],
                               ui[j,1,1], pivot='tail',linewidth=linewidth,scale=s)


    # grid plotting functions
    import itertools

    """
    Example usage:
    (grid,Nx,Ny)=getGrid(-1,1,-1,1,xpts=50,ypts=50)
    plotGrid(grid,Nx,Ny)
    """

    def d2zip(self,grid):
        return np.dstack(grid).reshape([-1,2])

    def d2unzip(self,points,Nx,Ny):
        return np.array([points[:,0].reshape(Nx,Ny),points[:,1].reshape(Nx,Ny)])

    def getGrid(self,xmin,xmax,ymin,ymax,xres=None,yres=None,xpts=None,ypts=None):
        """
        Make regular grid
        Grid spacing is determined either by (x|y)res or (x|y)pts
        """

        if xres:
            xd = xres
        elif xpts:
            xd = np.complex(0,xpts)
        else:
            assert(False)
        if yres:
            yd = yres
        elif ypts:
            yd = np.complex(0,ypts)
        else:
            assert(False)

        grid = np.mgrid[xmin:xmax:xd,ymin:ymax:yd]
        Nx = grid.shape[1]
        Ny = grid.shape[2]

        return (self.d2zip(grid),Nx,Ny)


    def plotGrid(self,grid,Nx,Ny,coloring=True):
        """
        Plot grid
        """

        xmin = grid[:,0].min(); xmax = grid[:,0].max()
        ymin = grid[:,1].min(); ymax = grid[:,1].max()
        border = .5*(0.2*(xmax-xmin)+0.2*(ymax-ymin))

        grid = self.d2unzip(grid,Nx,Ny)

        color = 0.75
        colorgrid = np.full([Nx,Ny],color)
        cm = plt.cm.get_cmap('gray')
        if coloring:
            cm = plt.cm.get_cmap('coolwarm')
            hx = (xmax-xmin) / (Nx-1)
            hy = (ymax-ymin) / (Ny-1)
            for i,j in itertools.product(range(Nx),range(Ny)):
                p = grid[:,i,j]
                xs = np.empty([0,2])
                ys = np.empty([0,2])
                if 0 < i:
                    xs = np.vstack((xs,grid[:,i,j]-grid[:,i-1,j],))
                if i < Nx-1:
                    xs = np.vstack((xs,grid[:,i+1,j]-grid[:,i,j],))
                if 0 < j:
                    ys = np.vstack((ys,grid[:,i,j]-grid[:,i,j-1],))
                if j < Ny-1:
                    ys = np.vstack((ys,grid[:,i,j+1]-grid[:,i,j],))

                Jx = np.mean(xs,0) / hx
                Jy = np.mean(ys,0) / hy
                J = np.vstack((Jx,Jy,)).T

                A = .5*(J+J.T)-np.eye(2)
                CSstrain = np.log(np.trace(A*A.T))
                logdetJac = np.log(sp.linalg.det(J))
                colorgrid[i,j] = logdetJac

            cmin = np.min(colorgrid)
            cmax = np.max(colorgrid)
            f = 2*np.max((np.abs(cmin),np.abs(cmax),.5))
            colorgrid = colorgrid / f + 0.5

            print("mean color: ", np.mean(colorgrid))

        # plot lines
        for i,j in itertools.product(range(Nx),range(Ny)):
            if i < Nx-1:
                plt.plot(grid[0,i:i+2,j],grid[1,i:i+2,j],color=cm(colorgrid[i,j]))
            if j < Ny-1:
                plt.plot(grid[0,i,j:j+2],grid[1,i,j:j+2],color=cm(colorgrid[i,j]))

        #for i in range(0,grid.shape[1]):
        #    plt.plot(grid[0,i,:],grid[1,i,:],color)
        ## plot x lines
        #for i in range(0,grid.shape[2]):
        #    plt.plot(grid[0,:,i],grid[1,:,i],color)


        plt.xlim(xmin-border,xmax+border)
        plt.ylim(ymin-border,ymax+border)


    ### Misc
    def ellipse(self, cent, Amp):
        return  np.vstack(( Amp[0]*np.cos(np.linspace(0,2*np.pi*(1-1./self.N.eval()),self.N.eval()))+cent[0], Amp[1]*np.sin(np.linspace(0,2*np.pi*(1-1./self.N.eval()),self.N.eval()))+cent[1] )).T


