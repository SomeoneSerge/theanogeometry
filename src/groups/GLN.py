from src.setup import *
from src.params import *
from src.group import *

##########################################################################
# this file contains definitions for G=GL(N)                             #
##########################################################################

group = 'GL(N)'

N = theano.shared(3) # N in GL(N)
G_dim = N*N # group dimension
G_emb_dim = N*N # matrix/embedding space dimension

## coordinate chart on the linking Lie algebra, trival in this case
def VtoLA(hatxi): # from \RR^G_dim to LA
    if hatxi.type == T.vector().type:
        return hatxi.reshape((N,N))
    else: # matrix
        return hatxi.reshape((N,N,-1))
def LAtoV(m): # from LA to \RR^G_dim
    if m.type == T.matrix().type:
        return m.reshape((G_dim,))
    elif m.type == T.tensor3().type:
        return m.reshape((G_dim,-1))
    else:
        assert(False)

#import theano.tensor.slinalg
Expm = T.slinalg.Expm()
#Expm = linalg.Expm()
Logm = lambda g : linalg.Logm()(g)

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
            plt.plot(ss[:,0,i],ss[:,1,i],ss[:,2,i],linewidth=.3,color=colors[i])
