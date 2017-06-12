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


#############################
# Setup for ellipsoid       #
#############################

manifold = 'ellipsoid'

d = T.constant(2)
M_emb_dim = T.constant(3)
#rank = T.constant(2) # does rank belong here?

# map F stereographic R2_r\rightarrow R3_s
q = T.vector()
zeroQ = T.zeros(d)
ellipsoid = theano.shared(np.array([1,1,1]))
F = lambda q: ellipsoid*T.stack([2*q[0],2*q[1],-(-1+q[0]**2+q[1]**2)])/(1+q[0]**2+q[1]**2)
Ff = theano.function([q], F(q))

JF = lambda q: T.jacobian(F(q),q)
JFf = theano.function([q], JF(q))

# metric matrix
gM = lambda q: T.dot(JF(q).T,JF(q))

# action of matrix group on elements
x = T.vector()
g = T.matrix() # \RR^{NxN} matrix
gs = T.tensor3() # sequence of \RR^{NxN} matrices
act = lambda g,x: T.tensordot(g,x,(1,0)) 
actf = theano.function([g,x], act(g,x))
actsf = theano.function([gs,x], act(gs,x))

########################
# Plotting:            #
########################
 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.ticker as ticker

def plotM(rotate=None, alpha = None,lw = 0.3):
    ax = plt.gca(projection='3d')
    x = np.arange(-10,10,1)
    ax.w_xaxis.set_major_locator(ticker.FixedLocator(x))
    ax.w_yaxis.set_major_locator(ticker.FixedLocator(x))
    ax.w_zaxis.set_major_locator(ticker.FixedLocator(x))
    ax.w_xaxis.set_pane_color((0.98, 0.98, 0.99, 1.0))
    ax.w_yaxis.set_pane_color((0.98, 0.98, 0.99, 1.0))
    ax.w_zaxis.set_pane_color((0.98, 0.98, 0.99, 1.0))
    ax.xaxis._axinfo["grid"]['linewidth'] = lw
    ax.yaxis._axinfo["grid"]['linewidth'] = lw
    ax.zaxis._axinfo["grid"]['linewidth'] = lw
#    ax.set_xlim(-1.5,1.5)
#    ax.set_ylim(-1.5,1.5)
#    ax.set_zlim(-1.5,1.5)
    ax.set_aspect("equal")
    if rotate != None:
        ax.view_init(rotate[0],rotate[1])
#     else:
#         ax.view_init(35,225)
    plt.xlabel('x')
    plt.ylabel('y')
    
#    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))
    #draw ellipsoid
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x=ellipsoid.get_value()[0]*np.cos(u)*np.sin(v)
    y=ellipsoid.get_value()[1]*np.sin(u)*np.sin(v)
    z=ellipsoid.get_value()[2]*np.cos(v)
    ax.plot_wireframe(x, y, z, color='gray', alpha=0.5)
    
    if alpha is not None:
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        x=ellipsoid.get_value()[0]*np.cos(u)*np.sin(v)
        y=ellipsoid.get_value()[1]*np.sin(u)*np.sin(v)
        z=ellipsoid.get_value()[2]*np.cos(v)
        ax.plot_surface(x, y, z, color=cm.jet(0.), alpha=alpha)

# plot x on ellipsoid. x can be either in coordinates or in R^3
def plotx(x,ui=None,color='b', color_intensity=1.,linewidth=1., s=15., prevx=None,last=True):
    if len(x.shape)>1:
        for i in range(x.shape[0]):
            plotx(x[i],ui=ui if i==0 else None,
                  color=color,
                  color_intensity=color_intensity if i==0 or i==x.shape[0]-1 else .7,
                  linewidth=linewidth,
                  s=s,
                  prevx=x[i-1] if i>0 else None,
                  last=i==(x.shape[0]-1))
        return

    xq = x
    if x.shape[0] < 3: # map to S2
        x = Ff(x)
        
    ax = plt.gca(projection='3d')
    if prevx is None or last:
        ax.scatter(x[0],x[1],x[2],color=color,s=s)
    if prevx is not None:
        if prevx.shape[0] < 3:
            prevx = Ff(prevx)
        xx = np.stack((prevx,x))
        ax.plot(xx[:,0],xx[:,1],xx[:,2],linewidth=linewidth,color=color)

    if ui is not None:
            JFgammai = JFf(xq)
            ui = np.dot(JFgammai,ui)
            ax.quiver(x[0],x[1],x[2],ui[0],ui[1],ui[2],
                      pivot='tail',
                      arrow_length_ratio = 0.15, linewidths=linewidth, length=0.5,
                      color='black')

# Plot of geodesic in R^2:
def plotR2x(x,ui=None,color='b',color_intensity=1.,linewidth=3.,prevx=None,last=True):
    if len(x.shape)>1:
        for i in range(x.shape[0]):
            plotR2x(x[i],ui=ui if i==0 else None,
                    color=color,
                    linewidth=linewidth if i==0 or i==x.shape[0]-1 else .8,
                    color_intensity=color_intensity if i==0 or i==x.shape[0]-1 else .7,
                    prevx=x[i-1] if i>0 else None,
                    last=i==(x.shape[0]-1))
        return
    
    if prevx is None or last:
        plt.scatter(x[0],x[1],color=color)
    if prevx is not None:
        xx = np.stack((prevx,x))
        plt.plot(xx[:,0],xx[:,1],linewidth=linewidth,color=color)

    if ui is not None:
        plt.quiver(x[0],x[1],ui[0],ui[1],pivot='tail',linewidth=linewidth,scale=5,color='black')

#### Geodesics (FM):
# Plot of geodesic in R^2:
def plotR2FMx(q,N_vec=None,i0=0,color='b',color_intensity=1.,linewidth=3.,prevx=None,last=True):
    if len(q.shape)>1:
        for i in range(q.shape[0]):
            plotR2FMx(q[i],
                      N_vec=N_vec,i0=i,
                      color=color,
                      linewidth=linewidth if i==0 or i==q.shape[0]-1 else .8,
                      color_intensity=color_intensity if i==0 or i==q.shape[0]-1 else .7,
                      prevx=q[i-1] if i>0 else None,
                      last=i==(q.shape[0]-1))
        return
    
    x = q[0:d.eval()]
    ui = q[d.eval():].reshape((d.eval(),2)) 
    
    if prevx is None or last:
        plt.scatter(x[0],x[1])
    if prevx is not None:
        prevxx = prevx[0:d.eval()]
        xx = np.stack((prevxx,x))
        plt.plot(xx[:,0],xx[:,1],linewidth=linewidth,color=color)
    
    # Frame along curve:
    if N_vec is not None:
        Seq = lambda m, n: [t*n//m + n//(2*m) for t in range(m)]
        Seqv = Seq(N_vec,n_steps.get_value())
        if i0 in Seqv:
            for j in range(d.eval()):
                plt.quiver(x[0],x[1],ui[0],ui[1], pivot='tail',
                           linewidth=linewidth,scale=5,color='black')

# Plot of geodesic Sphere
def plotFMx(q,N_vec=None,i0=0,color='b',color_intensity=1.,linewidth=3.,s=15.,prevx=None,last=True):
        if len(q.shape)>1:
            for i in range(q.shape[0]):
                plotFMx(q[i],
                      N_vec=N_vec,i0=i,
                      color=color,
                      linewidth=linewidth if i==0 or i==q.shape[0]-1 else .8,
                      color_intensity=color_intensity if i==0 or i==q.shape[0]-1 else .7,
                      prevx=q[i-1] if i>0 else None,
                      last=i==(q.shape[0]-1))
            return

        x = q[0:d.eval()]
        ui = q[d.eval():].reshape((d.eval(),2))
        
        xq = x
        if x.shape[0] < 3: # map to S2
            x = Ff(x)
         
        ax = plt.gca(projection='3d')
        if prevx is None or last:
            ax.scatter(x[0],x[1],x[2],color=color)
        if prevx is not None:
            prevxx = prevx[0:d.eval()]
            if prevxx.shape[0] < 3:
                prevxx = Ff(prevxx)
            xx = np.stack((prevxx,x))
            ax.plot(xx[:,0],xx[:,1],xx[:,2],linewidth=linewidth,color=color)
        
        # Frame along curve:
        if N_vec is not None:
            Seq = lambda m, n: [t*n//m + n//(2*m) for t in range(m)]
            Seqv = Seq(N_vec,n_steps.get_value())
            if i0 in Seqv:
                for j in range(d.eval()):
                        JFgammai = JFf(xq)
                        uiq = np.dot(JFgammai,ui[j,:])
                        ax.quiver(x[0],x[1],x[2],uiq[0],uiq[1],uiq[2], pivot='tail',
                              arrow_length_ratio = 0.15, linewidths=linewidth, length=0.5,
                              color='black')

### Plot of curvature:
def plotCur(q,v0,v1,color='b'):

        x = q[0:d.eval()]
        ui = q[d.eval():].reshape((d.eval(),2))
        
        xq = x
        if x.shape[0] < 3: # map to S2
            x = Ff(x)
         
        ax = plt.gca(projection='3d')
        ax.scatter(x[0],x[1],x[2],color=color)

        # Frame along curve:
        curm = np.tensordot(np.tensordot(R_uif(xq,ui), v0, axes = [0,0]), v1, axes = [0,0])
        for j in range(d.eval()):
            JFgammai = JFf(xq)
            uiq = np.dot(JFgammai,ui[j,:])
            curV = np.dot(JFgammai,curm[j,:])
            ax.quiver(x[0],x[1],x[2],uiq[0],uiq[1],uiq[2], pivot='tail',
                      arrow_length_ratio = 0.15, linewidths=1.5,
                      color='black',normalize=True,length=np.linalg.norm(uiq)/2)
            #end_Hvecq = (x + uiq/2)
            ax.quiver(x[0],x[1],x[2],
                      curV[0],curV[1],curV[2], pivot='tail',
                      arrow_length_ratio = 0.15, linewidths=2,
                      color='red',normalize=True,length=np.linalg.norm(uiq)/2)
