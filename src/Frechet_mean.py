from src.manifold import *
from src.metric import *
from src.Hamiltonian import *

def Frechet_mean(y,q0,p0,options=None):
    
    global steps
    steps = []
    steps.append(q0)

    def fopts(x):
        res = 0.
        grad = np.zeros(d.eval())
        N = y.shape[0]
        sol = mpu.pool.imap(lambda pars: (Logf(x,y[pars[0],:],p0)[0],),mpu.inputArgs(range(N)))
        res = list(sol)
        Logs = mpu.getRes(res,0)     

        res = (1./N)*np.sum(np.square(Logs))        
        grad = -(2./N)*np.sum(Logs,0)

        return (res,grad)

    def save_step(k):
        global steps
        steps.append(k)
   
    try:
        mpu.openPool()
        res = minimize(fopts, q0, method='BFGS', jac=True, options=options, callback=save_step)
    except:
        mpu.closePool()
        raise
    else:
        mpu.closePool()
    
    return(res.x,res.fun,np.array(steps))

