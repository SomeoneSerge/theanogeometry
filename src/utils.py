from src.setup import *
from src.params import *
try:
    from src.manifold import *
except NameError:
    pass
try:
    from src.group import *
except NameError:
    pass

#######################################################################
# various useful functions                                            #
#######################################################################

# numeric optimizer
def get_minimizer(f,method=None,options=None):
    x = T.vector()
    ff = theano.function([x],f(x))
    gradff = theano.function([x],T.grad(f(x),x))

    def fopt(x):
        return (ff(x),gradff(x))

    return (lambda x0: minimize(fopt, x0, method=method, jac=True, options=None))

# Integrator (non-stochastic)
def integrator(ode_f,method=default_method):
    
    # euler:
    def euler(*y):
        t = y[-2]
        x = y[-1]
        return (t+dt,x+dt*ode_f(*y))

    # Runge-kutta:
    def integrator_rk4(ode_f):
        def rk4(*y):
            t = y[-2]
            x = y[-1]
            k1 = ode_f(y[0:-2],t,x)
            k2 = ode_f(y[0:-2],t+dt/2,x + dt/2*k1)
            k3 = ode_f(y[0:-2],t+dt/2,x + dt/2*k2)
            k4 = ode_f(y[0:-2],t,x + dt*k3)
            return (t+dt,x + dt/6*(k1 + 2*k2 + 2*k3 + k4))

    if method == 'euler':
        return euler
    elif method == 'rk4':
        return rk4
    else:
        assert(False)

# return symbolic path given ode and integrator
def integrate(ode,x,*y):
    (cout, updates) = theano.scan(fn=integrator(ode),
            outputs_info=[T.constant(0.),x],
            sequences=[*y],
            n_steps=n_steps)
    return cout

# sde functions should return (det,sto,Sigma) where
# det is determinisitc part, sto is stochastic part,
# and Sigma stochastic generator (i.e. often sto=dot(Sigma,dW)

# standard noise realisations
srng = RandomStreams()#seed=42)
dWt = T.matrix() # n_steps x d or n_steps x G_dim
try:
    dWs = srng.normal((n_steps,d), std=np.sqrt(dt))
    dWsf = theano.function([],dWs)
except NameError:
    pass
try:
    dWsG = srng.normal((n_steps,G_dim), std=np.sqrt(dt))
    dWsGf = theano.function([],dWsG)
except NameError:
    pass

def integrator_stratonovich(sde_f):
    def euler_heun(dW,t,x,*ys):
        (detx, stox, X, *dys) = sde_f(dW,t,x,*ys)
        tx = x + stox
        ys_new = ()
        for (y,dy) in zip(ys,dys):
            ys_new = ys_new + (y+dt*dy,)
        return (t+dt,x + dt*detx + 0.5*(stox + sde_f(dW,t+dt,tx,*ys)[1]), *ys_new)

    return euler_heun

def integrator_ito(sde_f):
    def euler(dW,t,x,*ys):
        (detx, stox, X, *dys) = sde_f(dW,t,x,*ys)
        ys_new = ()
        for (y,dy) in zip(ys,dys):
            ys_new = ys_new + (y+dt*dy,)
        return (t+dt,x + dt*detx + stox, *ys_new)

    return euler

def integrate_sde(sde,integrator,x,dWt,*ys):
    (cout, updates) = theano.scan(fn=integrator(sde),
            outputs_info=[T.constant(0.),x, *ys],
            sequences=[dWt],
            n_steps=n_steps)
    return cout

# Gram-Schmidt:
def GramSchmidt_f(innerProd):
    def GS(Frame,q):
        
        if len(Frame.shape) == 1:
            gS = Frame/np.sqrt(innerProd(Frame,Frame,q))
        else:
            gS = np.zeros_like(Frame)
            for j in range(0,Frame.shape[1]):
                gS[:,j] = Frame[:,j]
                for i in range(0,j):
                    foo = innerProd(Frame[:,j],gS[:,i],q)/ innerProd(gS[:,i],gS[:,i],q)
                    gS[:,j] = gS[:,j] - foo*gS[:,i]
        
                gS[:,j] = gS[:,j]/np.sqrt(innerProd(gS[:,j],gS[:,j],q))

        return gS

    return GS



