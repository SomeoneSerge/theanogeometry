from src.setup import *
from src.params import *
from src.manifold import *
from src.metric import *
from src.utils import *

# Frame bundle FM

############################################
#####    Hamiltonian geodesics on FM    ####
############################################

##### Frame bundle cometric coordinate matrix:

def gFMsharp(u):
    
    x = u[0:d]
    ui = u[d:].reshape((d,rank))
    GamX = T.tensordot(Gamma_gM(x), ui, 
                       axes = [2,0]).dimshuffle(0,2,1)
    
    delta = T.eye(ui.shape[0],ui.shape[1])
    W = T.tensordot(ui, ui, axes = [1,1]) + lambdag0*gM(x)
    
    gij = W
    gijb = -T.tensordot(W, GamX, axes = [1,2])
    giaj = -T.tensordot(GamX, W, axes = [2,0])
    giajb = T.tensordot(T.tensordot(GamX, W, axes = [2,0]), 
                        GamX, axes = [2,2])

    return gij,gijb,giaj,giajb

##### Hamiltonian on FM based on the pseudo metric tensor: 
lambdag0 = 0

xi = T.vector()
xia = T.matrix()
def Hsplit(x,ui,xi,xia):
     
    GamX = T.tensordot(Gamma_gM(x), ui, 
                       axes = [2,0]).dimshuffle(0,2,1)
    
    delta = T.eye(ui.shape[0],ui.shape[1])
    W = T.tensordot(ui, ui, axes = [1,1]) + lambdag0*gM(x)
    
    gij = W
    gijb = -T.tensordot(W, GamX, axes = [1,2])
    giaj = -T.tensordot(GamX, W, axes = [2,0])
    giajb = T.tensordot(T.tensordot(GamX, W, axes = [2,0]), 
                        GamX, axes = [2,2])
    
    xigxi = T.dot(T.tensordot(xi, gij, axes = [0,0]), xi)
    xigxia = T.tensordot(T.tensordot(xi, gijb, axes = [0,0]), 
                         xia, axes = [[0,1],[0,1]])
    xiagxi = T.tensordot(T.tensordot(xi, giaj, axes = [0,2]), 
                         xia, axes = [[0,1],[0,1]])
    xiagxia = T.tensordot(T.tensordot(giajb, xia, axes = [[2,3],[0,1]]), 
                          xia, axes = [[0,1],[0,1]])
    
    return 0.5*(xigxi + xigxia + xiagxi + xiagxia)

Hfm = lambda q,p: Hsplit(q[0:d],q[d:(d+rank*d)].reshape((d,rank)),\
                         p[0:d],p[d:(d+rank*d)].reshape((d,rank)))
Hfmf = theano.function([q,p],Hfm(q,p))

##### Evolution equations:
dq = lambda q,p: T.grad(Hfm(q,p),p)
dp = lambda q,p: -T.grad(Hfm(q,p),q)
#dqfmf = theano.function([q,p], dq(q,p))
#dpfmf = theano.function([q,p], dp(q,p))

def ode_Hamfm(t,x): # Evolution equations at (p,q).
    dqt = dq(x[0],x[1])
    dpt = dp(x[0],x[1])
    return T.stack((dqt,dpt))
Hamfm = lambda q,p: integrate(ode_Hamfm,T.stack((q,p)))
Hamfmf = theano.function([q,p], Hamfm(q,p))

## Geodesic
Expfm = lambda q,p: Hamfm(q,p)[1][-1,0]
Exptfm = lambda q,p: Hamfm(q,p)[1][:,0].dimshuffle((1,0))
Expfmf = theano.function([q,p], Expfm(q,p))
Exptfmf = theano.function([q,p], Exptfm(q,p))

# Most Probable Path
loss = lambda q,q1,X: 1./d*T.sum((Expfm(q,gMflat(q,X))[0:d] - q1[0:d])**2)
lossf = theano.function([q,q1,X], loss(q,q1,X))

def Logfm(q1,q2):
    def fopts(v):
        y = lossf(q1,q2,v)
        return y

    res = minimize(fopts, np.zeros([d.eval()+rank.eval()*d.eval()]), 
                   method='CG', jac=False, options={'disp': False, 
                                                    'maxiter': 50})
    return res.x

#########################################################
#####    Development and horizontal vector fields    ####
#########################################################

## Horizontal vector fields:
def Hori(x,ui):
    
    # Contribution from the coordinate basis for x: 
    dx = ui
    # Contribution from the basis for Xa:
    dui = -T.tensordot(ui, T.tensordot(ui, Gamma_gM(x), axes = [0,2]), axes = [0,2])

    duiv = dui.reshape((ui.shape[1],dui.shape[1]*dui.shape[2]))

    return T.concatenate([dx,duiv.T], axis = 0)

Horif = theano.function([x,ui],Hori(x,ui))

