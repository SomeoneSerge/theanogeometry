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

from src.groups.group import *

import matplotlib.pyplot as plt

class SON(LieGroup):
    """ Special Orthogonal Group SO(N) """

    def __init__(self,N=3):
        LieGroup.__init__(self,N=N,invariance='left')

        self.dim = constant(N*(N-1)//2) # group dimension
        self.injectivity_radius = constant(2*np.pi)

        # project to group (here using QR factorization)
        def to_group(g):
            (q,r) = T.nlinalg.qr(g)
            return T.dot(q,T.nlinalg.AllocDiag()(T.nlinalg.ExtractDiag()(r)))
        g = self.element()
        self.to_groupf = theano.function([g],to_group(g))

        ## coordinate chart linking Lie algebra LA={A\in\RR^{NxN}|\trace{A}=0} and V=\RR^G_dim
        # derived from https://stackoverflow.com/questions/25326462/initializing-a-symmetric-theano-dmatrix-from-its-upper-triangle
        r = T.arange(N)
        tmp_mat = r[np.newaxis, :] + ((N * (N - 3)) // 2-(r * (r - 1)) // 2)[::-1,np.newaxis]
        triu_index_matrix = T.triu(tmp_mat+1)-T.diag(T.diagonal(tmp_mat+1))

        def VtoLA(hatxi): # from \RR^G_dim to LA
            if hatxi.type == T.vector().type:
                m = T.concatenate((T.zeros(1),hatxi))[triu_index_matrix]
                return m-m.T
            else: # matrix
                m = T.concatenate((T.zeros((1,hatxi.shape[1])),hatxi))[triu_index_matrix,:]
                return m-m.dimshuffle((1,0,2))
        self.VtoLA = VtoLA
        self.LAtoV = lambda m: m[T.triu(T.ones((N,N))-T.diag(T.ones(N))).nonzero()] # from LA to \RR^G_dim

        # triu_index_matrixf = theano.function([], triu_index_matrix)
        # print(triu_index_matrixf())


        #import theano.tensor.slinalg
        #Expm = T.slinalg.Expm()
        def Expm(g): # hardcoded for skew symmetric matrices to allow higher-order gradients
        #    (w,V) = T.nlinalg.Eigh()(T.constant(1.j)*g)
            (w,Vr,Vj) = linalg.skewEigh()(g)

            # direct
        #     w = -T.constant(1j)*w
        #     expm = T.real(T.tensordot(V,T.tensordot(T.diag(T.exp(w)),T.conj(V.T),(1,0)),(1,0)))

            # expanding exp
        #     expm = T.tensordot(V,T.tensordot(T.diag(T.cos(-w)+T.constant(1j)*T.sin(-w)),T.conj(V.T),(1,0)),(1,0))

            # only real computations
            #Vr = T.real(V)
            #Vj = T.imag(V)
            expm = (
        #         T.tensordot(V,T.tensordot(T.diag(T.cos(-w)),T.conj(V.T),(1,0)),(1,0))
                T.tensordot(Vr,T.tensordot(T.diag(T.cos(-w)),Vr.T,(1,0)),(1,0))
                +T.tensordot(Vj,T.tensordot(T.diag(T.cos(-w)),Vj.T,(1,0)),(1,0))
        #         +T.tensordot(V,T.tensordot(T.diag(T.constant(1j)*T.sin(-w)),T.conj(V.T),(1,0)),(1,0))
                +T.tensordot(Vr,T.tensordot(T.diag(T.sin(-w)),Vj.T,(1,0)),(1,0))
                -T.tensordot(Vj,T.tensordot(T.diag(T.sin(-w)),Vr.T,(1,0)),(1,0))
                )

            return expm
        self.Expm = Expm
        self.Logm = lambda g : linalg.Logm()(g)#to_group(g))

        super(SON,self).initialize()

    def __str__(self):
        return "SO(%d) (dimemsion %d)" % (self.N.eval(),self.dim.eval())

    ### plotting
    import matplotlib.pyplot as plt
    def plotg(self,g,color_intensity=1.,color=None,linewidth=3.,prevg=None):
        if len(g.shape)>2:
            for i in range(g.shape[0]):
                self.plotg(g[i],
                      linewidth=linewidth if i==0 or i==g.shape[0]-1 else .3,
                      color_intensity=color_intensity if i==0 or i==g.shape[0]-1 else .7,
                      prevg=g[i-1] if i>0 else None)
            return
        s0 = np.eye(3) # shape
        s = np.dot(g,s0) # rotated shape
        if prevg is not None:
            prevs = np.dot(prevg,s0)

        colors = color_intensity*np.array([[1,0,0],[0,1,0],[0,0,1]])
        for i in range(s.shape[1]):
            plt.quiver(0,0,0,s[0,i],s[1,i],s[2,i],pivot='tail',linewidth=linewidth,color=colors[i] if color is None else color,arrow_length_ratio=.15,length=1)
            if prevg is not None:
                ss = np.stack((prevs,s))
                ss = ss/np.linalg.norm(ss,axis=1)[:,np.newaxis,:]
                plt.plot(ss[:,0,i],ss[:,1,i],ss[:,2,i],linewidth=.3,color=colors[i])

## skew symmetric test
## np.random.seed(42)
#v0=np.array(np.random.rand(3))
#xiv0=VtoLAf(v0)
## xiv0=np.abs(xiv0)
#print(sp.linalg.expm(xiv0)-expf(xiv0))
#print(np.linalg.eigh(1j*xiv0))
## v = np.random.rand(3,3)
## v = v+v.T
#v = VtoLAf(np.random.rand(3))
#exporgf = theano.function([xi],T.slinalg.Expm()((xi-xi.T)/2))
#dexporgf = theano.function([xi],T.jacobian(T.slinalg.Expm()(xi).flatten(),xi).reshape((3,3,3,3)))
#dexpf = theano.function([xi],T.jacobian(exp(xi).flatten(),xi).reshape((3,3,3,3)))
#d2expf = theano.function([xi],T.jacobian(T.jacobian(exp(xi).flatten(),xi).flatten(),xi).reshape((3,3,3,3,3,3)))
#dexpxiv0 = np.tensordot(dexpf(xiv0),v,((2,3),(0,1)))
#dexporgxiv0 = np.tensordot(dexporgf(xiv0),v,((2,3),(0,1)))
#print(d2expf(xiv0).shape)
#print(dexporgxiv0.shape)
#print(dexpxiv0.shape)
#print(dexporgxiv0)
#print(dexpxiv0)
#print(dexporgxiv0-dexpxiv0)
#
#from scipy.optimize import approx_fprime
#findiff = np.zeros((3,3,3,3))
#f = lambda q: exporgf(.5*(q-q.T))
#for i in range(3):
#    for j in range(3):
#        findiff[i,j,:,:] = approx_fprime(xiv0.flatten(),lambda q: f(q.reshape((3,3)))[i,j],1e-5).reshape((3,3))
#print("finite diff comparison")
#print(findiff.shape)
#print(np.tensordot(findiff,v,((2,3),(0,1))))
#print(np.tensordot(findiff,v,((2,3),(0,1)))-dexpxiv0)

## checks and tests
#v0=np.array([1,0,0])
#w0=np.array([0,1,0])
#xiv0=VtoLAf(v0)
#xiw0=VtoLAf(w0)
#print(xiv0)
#print(xiw0)
#print(gVf(v0,v0))
#print(gVf(v0,w0))
#print(gLAf(xiv0,xiv0))
#x = expf(xiv0)
#print(x)
#print(to_groupf(x))
#w0x=dLf(x,e.eval(),xiw0)
#print(w0x)
#print(dLf(invf(x),x,w0x))
#print(invpbf(x,w0x))
#print(gpsif(v0+w0))
#print(cogpsif(v0+w0))
#print(eiLA.eval()[:,:,0])
#print(eiLA.eval()[:,:,1])
#print(bracketf(eiLA.eval()[:,:,0],eiLA.eval()[:,:,1]))
## print(Cf().shape)
## print(Cf()[:,:,0,1])
#print(C.eval().shape)
#print(C.eval()[:,:,0])
#for i in range(G_dim.eval()):
#    print(eiLA.eval()[:,:,i])

## exp/log checks
#x = expf(xiv0)
#print(x)
#print(logf(x))
#x = expf(zeroLA.eval())
#print(x)
#print(logf(x))
## xi0 = np.array([[  1.56920747e-15,  -3.14129382e+00,   6.00990706e-05],
##  [  3.14129382e+00,  -7.17942280e-16,   1.74848648e-04],
##  [ -6.00990706e-05,  -1.74848648e-04,   2.99144818e-21]])
#x = np.array([[  9.99999950e-01,   2.98825781e-04,  -1.11317090e-04],
#       [ -2.98821564e-04,   9.99999955e-01,   3.82801967e-05],
#       [  1.11328410e-04,  -3.82472624e-05,   9.99999993e-01]])
#print(x)
#print(logf(x))
#
#Logm_zeroest = lambda g : linalg.Logm(mode='zeroest',LAtoV=LAtoV,VtoLA=VtoLA)(g)
#log_zeroest = Logm_zeroest
#log_zeroestf = theano.function([g],log_zeroest(g))
#print("log_zeroest")
#print(log_zeroestf(x))
#Logm_nearest = lambda g,w: linalg.Logm(mode='nearest',LAtoV=LAtoV,VtoLA=VtoLA)(g,w)
#log_nearest = Logm_nearest
#log_nearestf = theano.function([g,w],log_nearest(g,w))
#print("log_nearest")
#print(log_nearestf(x,np.array([3*np.pi,0,0])))
#
## derivative checks
#x = expf(xiv0)
#print(x)
#print(sp.linalg.expm(xiv0))
#print(logf(x))
#dExpm = lambda xi: T.jacobian(T.slinalg.Expm()(xi).flatten(),xi)#.reshape(xi.shape+xi.shape)
#dLogm = lambda g: T.jacobian(log(g).flatten(),g)#.reshape(g.shape+g.shape)
#dExpmf = theano.function([xi],dExpm(xi))
#dLogmf = theano.function([g],dLogm(g))
#print(dExpmf(xiv0).shape)
## for i in range(3):
##     for j in range(3):
##         print(dExpmf(xiv0)[:,i,j].reshape((3,3)))
#print(dLogmf(x).shape)
#print(np.linalg.norm(np.dot(dLogmf(x).reshape((9,9)),dExpmf(xiv0).reshape((9,9)))-np.eye(9),np.inf))
#
## DLf = theano.function([g,h],dL(g,h))
## print(DLf(invf(x),x))
#if hatxi.type == T.vector().type:
#    print(True)
#hatXi = T.matrix()
#VtoLAff = theano.function([hatXi], VtoLA(hatXi))
#B=VtoLAff(np.eye(G_dim.eval()))
#print(B.shape)
#for i in range(G_dim.eval()):
#    print(B[:,:,i])
#Xi = T.tensor3()
#LAtoVff = theano.function([Xi], LAtoV(Xi))
#print(LAtoVff(B))
