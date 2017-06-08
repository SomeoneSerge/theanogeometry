from src.setup import *

###############################################################
# geodesic ODE integration                                    #
###############################################################

# Hamiltonian using psi map
def H(q,p):
    return 0.5*cogpsi(q,p,p)
Hf = function([qe,pe],H(qe,pe))
dq = lambda q,p: T.grad(H(q,p),p)
dp = lambda q,p: -T.grad(H(q,p),q)
dqf = function([qe,pe], dq(qe,pe))
dpf = function([qe,pe], dp(qe,pe))

def ode_f(x):
    
    dqt = dq(x[0],x[1])
    dpt = dp(x[0],x[1])
        
    return T.stack((dqt,dpt))

def euler(x,dt):    
    return x+dt*ode_f(x)

# create loop symbolic loop
def sim(g,p):
    (cout, updates) = theano.scan(fn=euler,
            outputs_info=[T.stack((g,p))],
            non_sequences=[dt],
            n_steps=n_steps)
    return cout[-1]

# compile it
simf = function(inputs=[g,p], 
        outputs=sim(g,p), 
        updates=updates)

## geodesics
Exp = lambda g,v: sim(g,flat(v))[0]
Exppsi = lambda hatm,v: sim(psi(hatm),flat(v))[0]
DExppsi = lambda hatm,v: T.stack((
    T.jacobian(sim(psi(hatm),flat(v))[0].flatten(),hatm).reshape(N,N,DIM),
    T.jacobian(sim(psi(hatm),flat(v))[0].flatten(),v).reshape(N,N,DIM)
    ))
loss = 1./MDIM*T.sum(T.sqr(Exp(g,v)-h))
dloss = T.grad(loss,v)
lossf = function(inputs=[x,h], 
        outputs=loss, 
        updates=updates)
dlossf = function(inputs=[x,h], 
        outputs=[loss, dloss], 
        updates=updates)

# shooting
from scipy.optimize import minimize,fmin_bfgs,fmin_cg

def shoot(g,h):
    def fopts(x):
        [y,gy] = dlossf(np.stack([q0,x.reshape([N.eval(),DIM])]).astype(theano.config.floatX))
        return (y,gy[1].flatten())
    
    res = minimize(fopts, p0.flatten(), method='L-BFGS-B', jac=True, options={'disp': False, 'maxiter': maxiter})
    
    return(res.x,res.fun)
Logf = lambda g,h: shoot(g,h)
Logpsif = lambda hatm,hata: shoot(psi(hatm),psi(hata))
