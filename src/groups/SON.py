from src.setup import *
from src.params import *
from src.group import *

##########################################################################
# this file contains definitions for G=SO(N)                             #
##########################################################################

group = 'SO(N)'

N = theano.shared(3) # N in SO(N)
G_dim = N*(N-1)//2 # group dimension
G_emb_dim = N*N # matrix/embedding space dimension
G_injectivity_radius = T.constant(2*np.pi)

# project to group (here using QR factorization)
def to_group(g):
    (q,r) = T.nlinalg.qr(g)
    return T.dot(q,T.nlinalg.AllocDiag()(T.nlinalg.ExtractDiag()(r)))
g = T.matrix()
to_groupf = theano.function([g],to_group(g))

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
LAtoV = lambda m: m[T.triu(T.ones((N,N))-T.diag(T.ones(N))).nonzero()] # from LA to \RR^G_dim

triu_index_matrixf = theano.function([], triu_index_matrix)
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

Logm = lambda g : linalg.Logm()(g)#to_group(g))

### plotting
import matplotlib.pyplot as plt
def plotg(g,color_intensity=1.,color=None,linewidth=3.,prevg=None):
    if len(g.shape)>2:
        for i in range(g.shape[0]):
            plotg(g[i],
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
