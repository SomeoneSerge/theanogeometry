try:
    from src.manifold import *
    from src.metric import *
    print("Using metric Hamiltonian for Hamiltonian dynamics")
except NameError:
    pass
try:
    from src.group import *
    from src.energy_group import *
    print("Using group Hamiltonian for Hamiltonian dynamics")
except NameError:
    pass
from src.utils import *

###############################################################
# geodesic integration, Hamiltonian form                      #
###############################################################

dq = lambda q,p: T.grad(H(q,p),p)
dp = lambda q,p: -T.grad(H(q,p),q)
#dqf = theano.function([q,p], dq(q,p))
#dpf = theano.function([q,p], dp(q,p))

def ode_Ham(t,x):
    dqt = dq(x[0],x[1])
    dpt = dp(x[0],x[1])
    return T.stack((dqt,dpt))
Ham = lambda q,p: integrate(ode_Ham,T.stack((q,p)))
Hamf = theano.function([q,p], Ham(q,p))

## Geodesic
Exp = lambda q,p: Ham(q,p)[1][-1,0]
Expt = lambda q,p: Ham(q,p)[1][:,0].dimshuffle((1,0))
Expf = theano.function([q,p], Exp(q,p))
Exptf = theano.function([q,p], Expt(q,p))

# Logarithmic map
loss = lambda q,p,x: 1./d.eval()*T.sum(T.sqr(Exp(q,p)-x))
lossf = theano.function([q,p,x], loss(q,p,x))

from scipy.optimize import minimize,fmin_bfgs,fmin_cg
def shoot(q1,q2,p0):
    def fopts(x):
        y = lossf(q1,x,q2).astype(theano.config.floatX)
        return y
    
    res = minimize(fopts, p0, method='L-BFGS-B', jac=False, options={'disp': False, 'maxiter': 100})
    
    return(res.x,res.fun)

Logf = lambda q1,q2,p0: shoot(q1,q2,p0)

# Group geodesics
try:
    Exppsi = lambda q,v: Ham(q,flatpsi(q,v))[1][-1,0]
    Exptpsi = lambda q,v: Ham(q,flatpsi(q,v))[1][:,0]
    DExppsi = lambda q,v: (
        T.jacobian(Ham(q,flatpsi(q,v))[1][-1,0].flatten(),q).reshape(N,N,G_dim),
        T.jacobian(Ham(q,flatpsi(q,v))[1][-1,0].flatten(),v).reshape(N,N,G_dim)
        )
    Exp = lambda g,vg: invtrns(g,psi(Ham(zeroV,LAtoV(invpb(g,vg)))[1][-1,0]))
    Expt = lambda g,vg: invtrns(g,psi(Ham(zeroV,LAtoV(invpb(g,vg)))[1][:,0].dimshuffle((1,0))))
    loss = 1./G_emb_dim*T.sum(T.sqr(Exp(g,vg)-h))
    losspsi = 1./G_emb_dim*T.sum(T.sqr(Exppsi(q,v)-h))
    dlosspsi = (T.grad(losspsi,q),T.grad(losspsi,v))
    Expf = theano.function([g,vg], Exp(g,vg))
    Exppsif = theano.function([q,v], Exppsi(q,v))
    Exptpsif = theano.function([q,v], Exptpsi(q,v))
    #lossf = theano.function([g,vg,h], loss)
    #losspsif = theano.function([q,v,h], losspsi)
    #dlosspsif = theano.function([q,v,h], [losspsi, dlosspsi[0], dlosspsi[1]])
except NameError:
    pass

#
##### Evolution equations:
#dq = lambda q,p: T.grad(H(q,p),p) # Evolution equation for point q in FM.
#dp = lambda q,p: -T.grad(H(q,p),q) # Evolution equation for covector p in FM.
#dqf = theano.function([q,p], dq(q,p))
#dpf = theano.function([q,p], dp(q,p))
#
#def ode_f(qp): # Evolution equations at (p,q).
#    dqt = dq(qp[0],qp[1])
#    dpt = dp(qp[0],qp[1])
#
#    return T.stack((dqt,dpt))
#ode_ff = theano.function([qp], ode_f(qp))
#
#(cout, updates) = theano.scan(fn=integrator(ode_f),
#                              outputs_info=[qp],
#                              n_steps=n_steps)
#
## Compile the Path Evolution:
#simf = function(inputs=[qp],
#                outputs=cout,
#                updates=updates)
#
##### Geodesics on M:
#def Geodesic(q0,p0):
#    gamma_t = simf(np.stack((q0,p0)))
#    return (gamma_t[:,0])
#
#
#def development(gamma0,q0):
#    gamma_t = simfdev(gamma0,q0)
#    return (gamma_t)
#
#

## shooting
#from scipy.optimize import minimize,fmin_bfgs,fmin_cg
#
#def shoot(g,h):
#    def fopts(x):
#        [y,gy] = dlossf(np.stack([q0,x.reshape([N.eval(),G_dim])]).astype(theano.config.floatX))
#        return (y,gy[1].flatten())
#    
#    res = minimize(fopts, p0.flatten(), method='L-BFGS-B', jac=True, options={'disp': False, 'maxiter': maxiter})
#    
#    return(res.x,res.fun)
#Logf = lambda g,h: shoot(g,h)
#Logpsif = lambda hatm,hata: shoot(psi(hatm),psi(hata))
