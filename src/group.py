from src.setup import *
from src.params import *
import theano.tensor.slinalg

###############################################################
# Lie group functions                                         #
###############################################################

import importlib
globals().update(importlib.import_module('src.groups.'+group).__dict__)
print("Group: ", group)

## initialization
def init(lN):
    N.set_value(int(lN))
    try:
        sigma.set_value(np.eye(G_dim.eval(),G_dim.eval()))
    except NameError:
        pass
    initC()

# elements
e = T.eye(N,N) # identity element
zeroV = T.zeros((G_dim,)) # zero element in V
zeroLA = T.zeros((N,N)) # zero element in LA
hatxi = T.vector() # \RR^G_dim vector
q = T.vector() # configuration, \RR^G_dim vector
xi = T.matrix() # matrix in LA
eta = T.matrix() # matrix in LA
alpha = T.matrix() # matrix in LA^*
beta = T.matrix() # matrix in LA^*
g = T.matrix() # \RR^{NxN} matrix
gs = T.tensor3() # sequence of \RR^{NxN} matrices
h = T.matrix() # \RR^{NxN} matrix
vg = T.matrix() # \RR^{NxN} tangent vector at g
wg = T.matrix() # \RR^{NxN} tangent vector at g
vh = T.matrix() # \RR^{NxN} tangent vector at h
w = T.vector() # \RR^G_dim tangent vector in coordinates
v = T.vector() # \RR^G_dim tangent vector in coordinates
pg = T.matrix() # \RR^{NxN} cotangent vector at g
ph = T.matrix() # \RR^{NxN} cotangent vector at h
p = T.vector() # \RR^G_dim cotangent vector in coordinates
pp = T.vector() # \RR^G_dim cotangent vector in coordinates
mu = T.vector() # \RR^G_dim LA cotangent vector in coordinates

# compile group specific chart functions
VtoLAf = theano.function([hatxi], VtoLA(hatxi))
LAtoVf = theano.function([xi], LAtoV(xi))
#print(VtoLAf(np.arange(G_dim.eval()).astype(np.float32)))
#print(LAtoVf(VtoLAf(np.arange(G_dim.eval()).astype(np.float32))))
#print(LAtoVf(VtoLAf(np.arange(G_dim.eval()).astype(np.float32))).shape)

## group operations
inv = lambda g: T.nlinalg.MatrixInverse()(g)
invf = theano.function([g],inv(g))

## group exp/log maps
exp = Expm
expf = theano.function([xi],exp(xi))
def expt(xi):
    (cout, updates) = theano.scan(fn=lambda t,x,dt,xi: (t+dt,exp(t*xi)),
        outputs_info=[T.constant(0.),e],
        non_sequences=[dt,xi],
        n_steps=n_steps)
    return cout
exptf = theano.function([xi],expt(xi))
log = Logm
logf = theano.function([g],log(g))

## shooting
#from scipy.optimize import minimize,fmin_bfgs,fmin_cg
#logm_loss = lambda xi: (1./2.)*T.sum(T.sqr(exp(xi)-h))
#dlogm_loss = lambda xi: T.grad((1./2.)*T.sum(T.sqr(exp(xi)-h)),xi)
#logm_lossf = theano.function([xi],logm_loss(xi))
#dlogm_lossf = theano.function([xi],(logm_loss(xi),dlogm_loss(xi)))
#def shoot_logm(g,h):
#    res = minimize(dlogm_lossf, np.zeros((G_dim,G_dim)), method='L-BFGS-B', jac=True, options={'disp': False, 'maxiter': maxiter})
#    
#    return(res.x,res.fun)
#logmf = lambda g,h: shoot_logm(g,h)
#logpsif = lambda xi,eta: shoot(psi(xi),psi(eta))

## Lie algebra
eiV = T.eye(G_dim) # standard basis for V
eiLA = VtoLA(eiV) # pushforward eiV basis for LA
#stdLA = T.eye(N*N,N*N).reshape((N,N,N*N)) # standard basis for \RR^{NxN}
#eijV = theano.shared(np.eye(G_dim.eval())) # standard basis for V
#eijLA = theano.shared(np.zeros((int(N.eval()),int(N.eval()),int(G_dim.eval())))) # eij in LA
def bracket(xi,eta): 
    if xi.type == T.matrix().type and eta.type == T.matrix().type:
        return T.tensordot(xi,eta,(1,0))-T.tensordot(eta,xi,(1,0))
    elif xi.type == T.tensor3().type and eta.type == T.tensor3().type:
        return T.tensordot(xi,eta,(1,0)).dimshuffle((0,2,1,3))-T.tensordot(eta,xi,(1,0)).dimshuffle((0,2,1,3))
    else:
        assert(False)
bracketf = theano.function([xi,eta],bracket(xi,eta))
#C = bracket(eiLA,eiLA) # structure constants, debug
#C = T.nlinalg.lstsq(eiLA.reshape((N*N*G_dim*G_dim,G_dim*G_dim*G_dim)),bracket(eiLA,eiLA).reshape((N*N*G_dim*G_dim))).reshape((G_dim,G_dim,G_dim)) # structure constants
#Cf = theano.function([],C)
C = theano.shared(np.zeros((G_dim.eval(),G_dim.eval(),G_dim.eval()))) # structure constants
#def initLABasis():
#    leijV = eijV.eval()
#    leijLA = np.zeros((N.eval(),N.eval(),G_dim.eval()))
#    for i in range(G_dim.eval()):
#        leijLA[:,:,i] = VtoLAf(leijV[:,i])
#    eijLA.set_value(leijLA)
def initC():
    lC = np.zeros((G_dim.eval(),G_dim.eval(),G_dim.eval()))
    for i in range(G_dim.eval()):
        for j in range(G_dim.eval()):
            xij = bracket(eiLA[:,:,i],eiLA[:,:,j])
            lC[i,j,:] = T.nlinalg.lstsq()(
                    eiLA.reshape((N*N,G_dim)),
                    xij.flatten(),
                    rcond=-1
                    )[0].eval()
            #lC[i,j,:] = np.linalg.lstsq(
            #        eiLA.eval().reshape(N.eval()*N.eval(), G_dim.eval()),
            #        xij.eval().reshape(N.eval()*N.eval())
            #        )[0]
    C.set_value(lC)


## surjective mapping \psi:\RR^G_dim\rightarrow G
psi = lambda hatxi: exp(VtoLA(hatxi))
def dpsi(hatxi,v):
    dpsi = T.jacobian(exp(VtoLA(hatxi)).flatten(),hatxi).reshape((N,N,G_dim))
    if v:
        return T.tensordot(dpsi,v,(2,0))
    return dpsi
psif = theano.function([hatxi],psi(hatxi))
dpsif = theano.function([hatxi,v],dpsi(hatxi,v))

## left/right translation
L = lambda g,h: T.tensordot(g,h,(1,0)) # left translation L_g(h)=gh
R = lambda g,h: T.tensordot(h,g,(1,0)) # right translation R_g(h)=hg
# pushforward of L/R of vh\in T_hG
#dL = lambda g,h,vh: theano.gradient.Rop(L(theano.gradient.disconnected_grad(g),h).flatten(),h,vh).reshape((N,N))
def dL(g,h,vh=None): 
    dL = T.jacobian(L(theano.gradient.disconnected_grad(g),h).flatten(),h).reshape((N,N,N,N))
    if vh:
        return T.tensordot(dL,vh,((2,3),(0,1)))
    return dL
#dR = lambda g,h,vh: theano.gradient.Rop(R(theano.gradient.disconnected_grad(g),h).flatten(),h,vh).reshape((N,N))
dR = lambda g,h,vh: T.tensordot(T.jacobian(R(theano.gradient.disconnected_grad(g),h).flatten(),h),vh,((1,2),(0,1))).reshape((N,N))
# pullback of L/R of vh\in T_h^*G
codL = lambda g,h,vh: dL(g,h,vh).T
codR = lambda g,h,vh: dR(g,h,vh).T
dLf = theano.function([g,h,vh],dL(g,h,vh))
dRf = theano.function([g,h,vh],dR(g,h,vh))
codLf = theano.function([g,h,ph],codL(g,h,ph))
codRf = theano.function([g,h,ph],codR(g,h,ph))

## actions
Ad = lambda g,xi: dR(inv(g),g,dL(g,e,xi))
ad = lambda xi,eta: bracket(xi,eta)
coad = lambda p,pp: T.tensordot(T.tensordot(C,p,(0,0)),pp,(1,0)) # TODO: check this
Adf = theano.function([g,xi],Ad(g,xi))
adf = theano.function([xi,eta],ad(xi,eta))
coadf = theano.function([p,pp],coad(p,pp))

## invariance
if invariance == 'left':
    invtrns = L # invariance translation
    invpb = lambda g,vg: dL(inv(g),g,vg) # left invariance pullback from TgG to LA
    invpf = lambda g,xi: dL(g,e,xi) # left invariance pushforward from LA to TgG
    invcopb = lambda g,pg: codL(inv(g),g,pg) # left invariance pullback from Tg^*G to LA^*
    invcopf = lambda g,alpha: codL(g,e,alpha) # left invariance pushforward from LA^* to Tg^*G
    infgen = lambda xi,g: dR(g,e,xi) # infinitesimal generator
else:
    invtrns = R # invariance translation
    invpb = lambda g,vg: dR(inv(g),g,vg) # right invariance pullback from TgG to LA
    invpf = lambda g,xi: dR(g,e,xi) # right invariance pushforward from LA to TgG
    invcopb = lambda g,pg: codR(inv(g),g,pg) # right invariance pullback from Tg^*G to LA^*
    invcopf = lambda g,alpha: codR(g,e,alpha) # right invariance pushforward from LA^* to Tg^*G
    infgen = lambda xi,g: dL(g,e,xi) # infinitesimal generator
invpbf = theano.function([g,vg],invpb(g,vg))
invpff = theano.function([g,xi],invpf(g,xi))
invcopbf = theano.function([g,pg],invcopb(g,pg))
invcopff = theano.function([g,alpha],invcopf(g,alpha))
infgenf = theano.function([xi,g],infgen(xi,g))

