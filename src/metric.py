from src.setup import *
from src.utils import *
try:
# Riemannian metric and Hamiltonian on M
    from src.manifold import *
    
    try:
        gMq = gM(q) # metric
    except NameError:
        miss = 1
    
    try:
        gMsharpq = gMsharp(q) # sharp map (cometric)
    except NameError:
        miss = 2
    
    if miss == 2:
        gMsharp = lambda q: T.nlinalg.matrix_inverse(gM(q))
    if miss == 1:
        gM = lambda q: T.nlinalg.matrix_inverse(gMsharp(q))
    
    gMf = theano.function([q],gM(q))
    gMsharpf = theano.function([q],gMsharp(q))
    
    DgM = lambda q: T.jacobian(gM(q).flatten(),q).reshape((d,d,d)) # Derivative of metric
    DgMf = theano.function([q],DgM(q))

    ##### Measure
    muM_Q = lambda q: 1./T.nlinalg.Det()(gM(q))
    muM_Qf = theano.function([q],muM_Q(q))
    
    ##### Sharp and flat map:
    DgMsharp = lambda q: T.jacobian(gMsharp(q).flatten(),q).reshape((d,d,d)) # Derivative of sharp map
    
    gMflat = lambda q,X: T.dot(gM(q),X)
    gMflatf = theano.function([q,X], gMflat(q,X))
    
    ##### Christoffel symbols
    Gamma_gM = lambda q: 0.5*(T.tensordot(gMsharp(q),DgM(q),axes = [1,0])\
                   +T.tensordot(gMsharp(q),DgM(q),axes = [1,0]).dimshuffle(0,2,1)\
                   -T.tensordot(gMsharp(q),DgM(q),axes = [1,2]))
    Gamma_gMf = theano.function([q],Gamma_gM(q))
    
    # Inner Product on M:
    def innerProd(u1,u2,q):
    
        return np.dot(np.tensordot(gMf(q), u1, axes = [0,0]), u2)

    def GramSchmidt(u,q):
        return (GramSchmidt_f(innerProd))(u,q)

    ##### Hamiltonian
    H = lambda q,p: 0.5*T.dot(p,T.dot(gMsharp(q),p))
    Hf = theano.function([q,p],H(q,p))
except NameError:
    pass

try:
    ## invariant group metric
    from src.group import *

    sigma = theano.shared(np.eye(G_dim.eval(),G_dim.eval())) # diffusion field
    sqrtA = inv(sigma) # square root metric
    A = T.tensordot(sqrtA,sqrtA,(0,0)) # metric
    W = inv(A) # covariance (cometric)
    def gV(v=None,w=None): 
        if not v and not w:
            return A
        elif v and not w:
            return T.tensordot(A,v,(1,0))
        elif v.type == T.vector().type and w.type == T.vector().type:
            return T.dot(v,T.dot(A,w))
        elif v.type == T.vector().type and not w:
            return T.dot(A,v)
        elif v.type == T.matrix().type and w.type == T.matrix().type:
            return T.tensordot(v,T.tensordot(A,w,(1,0)),(0,0))
        else:
            assert(False)
    def cogV(cov=None,cow=None): 
        if not cov and not cow:
            return W
        elif cov and not cow:
            return T.tensordot(W,cov,(1,0))
        elif cov.type == T.vector().type and cow.type == T.vector().type:
            return T.dot(cov,T.dot(W,cow))
        elif cov.type == T.matrix().type and cow.type == T.matrix().type:
            return T.tensordot(cov,T.tensordot(W,cow,(1,0)),(0,0))
        else:
            assert(False)
    def gLA(xiv,xiw): 
        v = LAtoV(xiv)
        w = LAtoV(xiw)
        return gV(v,w)
    def cogLA(coxiv,coxiw): 
        cov = LAtoV(coxiv)
        cow = LAtoV(coxiw)
        return cogV(cov,cow)
    def gG(g,vg,wg): 
        xiv = invpb(g,vg)
        xiw = invpb(g,wg)
        return gLA(xiv,xiw)
    def gpsi(hatxi,v=None,w=None): 
        g = psi(hatxi)
        vg = dpsi(hatxi,v)
        wg = dpsi(hatxi,w)
        return gG(g,vg,wg)
    def cogpsi(hatxi,p=None,pp=None): 
        invgpsi = inv(gpsi(hatxi))
        if p and pp:
            return T.tensordot(p,T.tensordot(invgpsi,pp,(1,0)),(0,0))
        elif p and not pp:
            return T.tensordot(invgpsi,p,(1,0))
        return invgpsi
    gGf = theano.function([g,vg,wg],gG(g,vg,wg))
    gpsi_evf = theano.function([hatxi,v,w],gpsi(hatxi,v,w))
    gpsif = theano.function([hatxi],gpsi(hatxi))
    cogpsi_evf = theano.function([hatxi,p,pp],cogpsi(hatxi,p,pp))
    cogpsif = theano.function([hatxi],cogpsi(hatxi))
    xiv = T.matrix() # matrix in LA
    xiw = T.matrix() # matrix in LA
    gLAf = theano.function([xiv,xiw],gLA(xiv,xiw))
    gVf = theano.function([v,w],gV(v,w))
    
    # sharp/flat mappings
    def sharpV(mu):
        return T.dot(W,mu)
    def flatV(v):
        return T.dot(A,v)
    def sharp(g,pg):
        return invpf(g,VtoLA(T.dot(W,LAtoV(invcopb(g,pg)))))
    def flat(g,vg):
        return invcopf(g,VtoLA(T.dot(A,LAtoV(invpb(g,vg)))))
    def sharppsi(hatxi,p):
        return T.tensordot(cogpsi(hatxi),p,(1,0))
    def flatpsi(hatxi,v):
        return T.tensordot(gpsi(hatxi),v,(1,0))
    sharpVf = theano.function([mu],sharpV(mu))
    flatVf = theano.function([v],flatV(v))
    sharpf = theano.function([g,pg],sharp(g,pg))
    flatf = theano.function([g,vg],flat(g,vg))
    sharppsif = theano.function([hatxi,p],sharppsi(hatxi,p))
    flatpsif = theano.function([hatxi,v],flatpsi(hatxi,v))
except NameError:
    pass
