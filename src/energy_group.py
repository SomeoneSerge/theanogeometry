from src.setup import *
from src.group import *
from src.metric import *

# group Lagrangian and Hamiltonian

# Lagrangian
def L(g,vg):
    return .5*gG(g,vg,vg)
Lf = theano.function([g,vg],L(g,vg))
# Lagrangian using psi map
def Lpsi(q,v):
    return .5*gpsi(q,v,v)
dLpsidq = lambda q,v: T.grad(Lpsi(q,v),q)
dLpsidv = lambda q,v: T.grad(Lpsi(q,v),v)
# LA restricted Lagrangian
def l(hatxi):
    return 0.5*gV(hatxi,hatxi)
dldhatxi = lambda hatxi: T.grad(l(hatxi),hatxi)
Lpsif = theano.function([q,v],Lpsi(q,v))
lf = theano.function([hatxi],l(hatxi))

# Hamiltonian using psi map
def Hpsi(q,p):
    return .5*cogpsi(q,p,p)
# LA^* restricted Hamiltonian
def Hminus(mu):
    return .5*cogV(mu,mu)
dHminusdmu = lambda mu: T.grad(Hminus(mu),mu)
Hpsif = theano.function([q,p],Hpsi(q,p))
Hminusf = theano.function([mu],Hminus(mu))

# Legendre transformation. The above Lagrangian is hyperregular
FLpsi = lambda q,v: (q,dLpsidv(q,v))
invFLpsi = lambda q,p: (q,cogpsi(q,p))
def HL(q,p): 
    (q,v) = invFLpsi(q,p)
    return T.dot(p,v)-L(q,v)
Fl = lambda hatxi: dldhatxi(hatxi)
invFl = lambda mu: cogV(mu)
def hl(mu):
    hatxi = invFl(mu)
    return T.dot(mu,hatxi)-l(hatxi)
FLpsif = theano.function([q,v],FLpsi(q,v))
invFLpsif = theano.function([q,p],invFLpsi(q,p))
Flf = theano.function([hatxi],Fl(hatxi))
invFlf = theano.function([mu],invFl(mu))

# default Hamiltonian
H = Hpsi

# A.set_value(np.diag([3,2,1]))
# print(FLpsif(q0,v0))
# print(invFLpsif(q0,p0))
# (flq0,flv0)=FLpsif(q0,v0)
# print(q0,v0)
# print(invFLpsif(flq0,flv0))
