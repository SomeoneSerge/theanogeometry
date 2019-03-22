from __future__ import print_function

import logging
import warnings

import numpy
import scipy
from six.moves import xrange
from scipy.optimize import minimize

import theano
from theano.tensor import as_tensor_variable
from theano.gof import Op, Apply
from theano.gradient import DisconnectedType
#from teano.tensor import basic as tensor
from theano import tensor as T

logger = logging.getLogger(__name__)

class symEigh(theano.tensor.nlinalg.Eig):
    """
    Return the eigenvalues and eigenvectors of a symmetric matrix.
    Returning symbolic gradient allowing automatic higher-order derivatives.
    """

    _numop = staticmethod(numpy.linalg.eigh)

    def __init__(self):
        None

    def make_node(self, x):
        x = as_tensor_variable(x)
        assert x.ndim == 2
        # Numpy's linalg.eigh may return either double or single
        # presision eigenvalues depending on installed version of
        # LAPACK.  Rather than trying to reproduce the (rather
        # involved) logic, we just probe linalg.eigh with a trivial
        # input.
        w_dtype = self._numop([[numpy.dtype(x.dtype).type()]])[0].dtype.name
        w = theano.tensor.vector(dtype=w_dtype)
        v = theano.tensor.matrix(dtype=x.dtype)
        return Apply(self, [x], [w, v])

    def perform(self, node, inputs, outputs):
        (x,) = inputs
        (w, v) = outputs
        x = (x+x.T)/2
        w[0], v[0] = self._numop(x)

    def infer_shape(self, node, shapes):
        n = shapes[0][0]
        return [(n,), (n, n)]

    def grad(self, inputs, g_outputs):
        r"""The gradient function should return
           .. math:: \sum_n\left(W_n\frac{\partial\,w_n}
                           {\partial a_{ij}} +
                     \sum_k V_{nk}\frac{\partial\,v_{nk}}
                           {\partial a_{ij}}\right),
        where [:math:`W`, :math:`V`] corresponds to ``g_outputs``,
        :math:`a` to ``inputs``, and  :math:`(w, v)=\mbox{eig}(a)`.
        Analytic formulae for eigensystem gradients are well-known in
        perturbation theory:
           .. math:: \frac{\partial\,w_n}
                          {\partial a_{ij}} = v_{in}\,v_{jn}
           .. math:: \frac{\partial\,v_{kn}}
                          {\partial a_{ij}} =
                \sum_{m\ne n}\frac{v_{km}v_{jn}}{w_n-w_m}
                
        Code derived from theano.nlinalg.Eigh and doi=10.1.1.192.9105
        """
        x, = inputs
        w, v = self(x)
        # Replace gradients wrt disconnected variables with
        # zeros. This is a work-around for issue #1063.
        W, V = _zero_disconnected([w, v], g_outputs)
        
        N = x.shape[0]

        # W part
        gW = T.tensordot(v,v*W[numpy.newaxis,:],(1,1))
        # V part
        vv = v[:,:,numpy.newaxis,numpy.newaxis]*v[numpy.newaxis,numpy.newaxis,:,:]
        minusww = -w[:,numpy.newaxis]+w[numpy.newaxis,:]
        minuswwinv = 1/(minusww+T.eye(N))
        minuswwinv = T.triu(minuswwinv,1)+T.tril(minuswwinv,-1)# remove diagonal
        c = (vv*minuswwinv[numpy.newaxis,:,numpy.newaxis,:]).dimshuffle((1,3,0,2))
        vc = T.tensordot(v,c,(1,0))
        gV = T.tensordot(V,vc,((0,1),(0,1)))

        g = gW+gV
        
        res = (g.T+g)/2
        return [res]

class symEighSqrt(theano.tensor.nlinalg.Eig):
    """
    Return the eigenvalues and eigenvectors of a symmetric matrix A given by its square root B, i.e. A=BB^H
    Returning symbolic gradient allowing automatic higher-order derivatives.
    """

    _numop = staticmethod(numpy.linalg.svd)

    def __init__(self):
        None

    def make_node(self, x):
        x = as_tensor_variable(x)
        assert x.ndim == 2
        # Numpy's linalg.eigh may return either double or single
        # presision eigenvalues depending on installed version of
        # LAPACK.  Rather than trying to reproduce the (rather
        # involved) logic, we just probe linalg.eigh with a trivial
        # input.
        w_dtype = self._numop([[numpy.dtype(x.dtype).type()]])[0].dtype.name
        w = theano.tensor.vector(dtype=w_dtype)
        v = theano.tensor.matrix(dtype=x.dtype)
        return Apply(self, [x], [w, v])

    def perform(self, node, inputs, outputs):
        (x,) = inputs
        (w, v) = outputs
        u, s, vh = self._numop(x, full_matrices=False)
        w[0], v[0] = s ** 2, u

    def infer_shape(self, node, shapes):
        n = shapes[0][0]
        d = shapes[0][1]
        return [(d,), (n, d)]

    def grad(self, inputs, g_outputs):
        r"""The gradient function should return
           .. math:: \sum_n\left(W_n\frac{\partial\,w_n}
                           {\partial a_{ij}} +
                     \sum_k V_{nk}\frac{\partial\,v_{nk}}
                           {\partial a_{ij}}\right),
        where [:math:`W`, :math:`V`] corresponds to ``g_outputs``,
        :math:`a` to ``inputs``, and  :math:`(w, v)=\mbox{eig}(a)`.
        Analytic formulae for eigensystem gradients are well-known in
        perturbation theory:
           .. math:: \frac{\partial\,w_n}
                          {\partial a_{ij}} = v_{in}\,v_{jn}
           .. math:: \frac{\partial\,v_{kn}}
                          {\partial a_{ij}} =
                \sum_{m\ne n}\frac{v_{km}v_{jn}}{w_n-w_m}

        Code derived from theano.nlinalg.Eigh and doi=10.1.1.192.9105
        """
        x, = inputs
        w, v = self(x)
        # Replace gradients wrt disconnected variables with
        # zeros. This is a work-around for issue #1063.
        W, V = _zero_disconnected([w, v], g_outputs)

        N = x.shape[0]

        # W part
        gW = T.tensordot(v, v * W[numpy.newaxis, :], (1, 1))
        # V part
        vv = v[:, :, numpy.newaxis, numpy.newaxis] * v[numpy.newaxis, numpy.newaxis, :, :]
        minusww = -w[:, numpy.newaxis] + w[numpy.newaxis, :]
        minuswwinv = 1 / (minusww + T.eye(N))
        minuswwinv = T.triu(minuswwinv, 1) + T.tril(minuswwinv, -1)  # remove diagonal
        c = (vv * minuswwinv[numpy.newaxis, :, numpy.newaxis, :]).dimshuffle((1, 3, 0, 2))
        vc = T.tensordot(v, c, (1, 0))
        gV = T.tensordot(V, vc, ((0, 1), (0, 1)))

        g = gW + gV

        res = (g.T + g) / 2
        return [res]


class skewEigh(theano.tensor.nlinalg.Eig):
    """
    Return the eigenvalues and eigenvectors of a skew symmetric matrix.
    Using only real computations and returning symbolic gradient allowing 
    automatic higher-order derivatives.
    """

    _numop = staticmethod(numpy.linalg.eigh)

    def __init__(self):
        None

    def make_node(self, x):
        x = as_tensor_variable(x)
        assert x.ndim == 2
        # Numpy's linalg.eigh may return either double or single
        # presision eigenvalues depending on installed version of
        # LAPACK.  Rather than trying to reproduce the (rather
        # involved) logic, we just probe linalg.eigh with a trivial
        # input.
        w_dtype = self._numop([[numpy.dtype(x.dtype).type()]])[0].dtype.name
        w = theano.tensor.vector(dtype=w_dtype)
        vr = theano.tensor.matrix(dtype=x.dtype)
        vj = theano.tensor.matrix(dtype=x.dtype)
        return Apply(self, [x], [w, vr, vj])

    def perform(self, node, inputs, outputs):
        (x,) = inputs
        (w, vr, vj) = outputs
        x = 1j*(x-x.T)/2
        w[0], v = self._numop(x)
        vr[0] = numpy.real(v)
        vj[0] = numpy.imag(v)

    def infer_shape(self, node, shapes):
        n = shapes[0][0]
        return [(n,), (n, n), (n, n)]

    def grad(self, inputs, g_outputs):
        r"""The gradient function should return
           .. math:: \sum_n\left(W_n\frac{\partial\,w_n}
                           {\partial a_{ij}} +
                     \sum_k V_{nk}\frac{\partial\,v_{nk}}
                           {\partial a_{ij}}\right),
        where [:math:`W`, :math:`V`] corresponds to ``g_outputs``,
        :math:`a` to ``inputs``, and  :math:`(w, v)=\mbox{eig}(a)`.
        Analytic formulae for eigensystem gradients are well-known in
        perturbation theory:
           .. math:: \frac{\partial\,w_n}
                          {\partial a_{ij}} = v_{in}\,v_{jn}
           .. math:: \frac{\partial\,v_{kn}}
                          {\partial a_{ij}} =
                \sum_{m\ne n}\frac{v_{km}v_{jn}}{w_n-w_m}
                
        Code derived from theano.nlinalg.Eigh and doi=10.1.1.192.9105
        """
        x, = inputs
        w, vr, vj = self(x)
        # Replace gradients wrt disconnected variables with
        # zeros. This is a work-around for issue #1063.
        W, Vr, Vj = _zero_disconnected([w, vr, vj], g_outputs)
        
#         # complex version
#         v = vr+1j*vj
#         V = Vr+1j*Vj
#         N = x.shape[0]

#         gW = T.tensordot(T.conj(v),v*W[numpy.newaxis,:],(1,1)) # W part
#         vv = T.conj(v[:,:,numpy.newaxis,numpy.newaxis])*v[numpy.newaxis,numpy.newaxis,:,:]
#         minusww = -w[:,numpy.newaxis]+w[numpy.newaxis,:]
#         minuswwinv = 1/(minusww+T.eye(N))
#         minuswwinv = T.triu(minuswwinv,1)+T.tril(minuswwinv,-1)# remove diagonal
#         c = (vv*minuswwinv[numpy.newaxis,:,numpy.newaxis,:]).dimshuffle((1,3,0,2))
#         vc = T.tensordot(v,c,(1,0))
#         gV = T.tensordot(T.conj(V),vc,((0,1),(0,1)))        
#         g = gW+gV

#         g = T.imag(g)
        
        # real version
        v = vr+1j*vj
        V = Vr+1j*Vj
        N = x.shape[0]

        # W part
        gWr = (T.tensordot(vr,vr*W[numpy.newaxis,:],(1,1))
              +T.tensordot(vj,vj*W[numpy.newaxis,:],(1,1)))
        gWj = (T.tensordot(vr,vj*W[numpy.newaxis,:],(1,1))
              -T.tensordot(vj,vr*W[numpy.newaxis,:],(1,1)))
        # V part
        vvr = (vr[:,:,numpy.newaxis,numpy.newaxis]*vr[numpy.newaxis,numpy.newaxis,:,:]
               +vj[:,:,numpy.newaxis,numpy.newaxis]*vj[numpy.newaxis,numpy.newaxis,:,:])
        vvj = (vr[:,:,numpy.newaxis,numpy.newaxis]*vj[numpy.newaxis,numpy.newaxis,:,:]
               -vj[:,:,numpy.newaxis,numpy.newaxis]*vr[numpy.newaxis,numpy.newaxis,:,:])
        minusww = -w[:,numpy.newaxis]+w[numpy.newaxis,:]
        minuswwinv = 1/(minusww+T.eye(N))
        minuswwinv = T.triu(minuswwinv,1)+T.tril(minuswwinv,-1)# remove diagonal
        cr = (vvr*minuswwinv[numpy.newaxis,:,numpy.newaxis,:]).dimshuffle((1,3,0,2))
        cj = (vvj*minuswwinv[numpy.newaxis,:,numpy.newaxis,:]).dimshuffle((1,3,0,2))
        vcr = (T.tensordot(vr,cr,(1,0))-T.tensordot(vj,cj,(1,0)))
        vcj = (T.tensordot(vr,cj,(1,0))+T.tensordot(vj,cr,(1,0)))
        gVr = (T.tensordot(Vr,vcr,((0,1),(0,1)))+T.tensordot(Vj,vcj,((0,1),(0,1))))
        gVj = (T.tensordot(Vr,vcj,((0,1),(0,1)))-T.tensordot(Vj,vcr,((0,1),(0,1))))

        g = gWj+gVj
        
        res = (g.T-g)/2
        return [res]

    
def _zero_disconnected(outputs, grads):
    l = []
    for o, g in zip(outputs, grads):
        if isinstance(g.type, DisconnectedType):
            l.append(o.zeros_like())
        else:
            l.append(g)
    return l

# The code is adopted from https://github.com/Theano/Theano/pull/3959
class LogAbsDet(Op):
    """Computes the logarithm of absolute determinant of a square
    matrix M, log(abs(det(M))), on CPU. Avoids det(M) overflow/
    underflow.
    TODO: add GPU code!
    """
    def make_node(self, x):
        x = theano.tensor.as_tensor_variable(x)
        o = theano.tensor.scalar(dtype=x.dtype)
        return Apply(self, [x], [o])

    def perform(self, node, inputs, outputs):
        try:
            (x,) = inputs
            (z,) = outputs
            s = numpy.linalg.svd(x, compute_uv=False)
            log_abs_det = numpy.sum(numpy.log(numpy.abs(s)))
            z[0] = numpy.asarray(log_abs_det, dtype=x.dtype)
        except Exception:
            print('Failed to compute logabsdet of {}.'.format(x))
            raise

    def grad(self, inputs, g_outputs):
        gz, = g_outputs
        x, = inputs
        return [gz * T.nlinalg.matrix_inverse(x).T]

    def __str__(self):
        return "LogAbsDet"

logabsdet = LogAbsDet()


#class Expm(Op):
#    """
#    Compute the matrix exponential of a square array.
#    """
#
#    __props__ = ()
#
#    def make_node(self, A):
#        A = as_tensor_variable(A)
#        assert A.ndim == 2
#        expm = theano.tensor.matrix(dtype=A.dtype)
#        return Apply(self, [A, ], [expm, ])
#
#    def perform(self, node, inputs, outputs):
#        (A,) = inputs
#        (expm,) = outputs
#        expm[0] = scipy.linalg.expm(A)
#
#    def grad(self, inputs, outputs):
#        (A,) = inputs
#        (g_out,) = outputs
#        return [ExpmGrad()(A, g_out)]
#
#    def infer_shape(self, node, shapes):
#        return [shapes[0]]
#
#
#class ExpmGrad(Op):
#    """
#    Gradient of the matrix exponential of a square array.
#    """
#
#    __props__ = ()
#
#    def make_node(self, A, gw):
#        A = as_tensor_variable(A)
#        assert A.ndim == 2
#        out = theano.tensor.matrix(dtype=A.dtype)
#        return Apply(self, [A, gw], [out, ])
#
#    def infer_shape(self, node, shapes):
#        return [shapes[0]]
#
#    def perform(self, node, inputs, outputs):
#        # Kalbfleisch and Lawless, J. Am. Stat. Assoc. 80 (1985) Equation 3.4
#        # Kind of... You need to do some algebra from there to arrive at
#        # this expression.
#        (A, gA) = inputs
#        (out,) = outputs
#        w, V = scipy.linalg.eig(A, right=True)
#        U = scipy.linalg.inv(V).T
#
#        exp_w = numpy.exp(w)
#        X = numpy.subtract.outer(exp_w, exp_w) / numpy.subtract.outer(w, w)
#        numpy.fill_diagonal(X, exp_w)
#        Y = U.dot(V.T.dot(gA).dot(U) * X).dot(V.T)
#
#        with warnings.catch_warnings():
#            warnings.simplefilter("ignore", numpy.ComplexWarning)
#            out[0] = Y.astype(A.dtype)

class Logm(Op):
    """
    Compute the matrix logarithm of a square array.
    """

    __props__ = ()
    __props__ = ('mode','exp','LAtoVf','VtoLAf','lossf','dlossf')

    def __init__(self, mode='matrix', exp=None, LAtoV=None, VtoLA=None):
        assert mode in ['matrix', 'zeroest', 'nearest']
        self.mode = mode
        if exp is None:
            exp = T.slinalg.Expm()
        self.exp = exp
        self.LAtoVf = None
        self.VtoLAf = None
        self.lossf = None
        self.dlossf = None
        if mode != 'matrix':
            g = T.matrix()
            hatxi = T.vector()
            xi = T.matrix()
            self.LAtoVf = theano.function([xi],LAtoV(xi))
            self.VtoLAf = theano.function([hatxi],VtoLA(hatxi))
            loss = lambda hatxi,g: T.sum((exp(VtoLA(hatxi))-g)**2)
            dloss = lambda hatxi,g: T.jacobian(loss(hatxi,g),hatxi)
            self.lossf = theano.function([hatxi,g],loss(hatxi,g))
            self.dlossf = theano.function([hatxi,g],dloss(hatxi,g)) 

    def make_node(self, A, w=None):
        A = as_tensor_variable(A)
        assert A.ndim == 2
        expm = theano.tensor.matrix(dtype=A.dtype)
        if self.mode != 'nearest':
            return Apply(self, [A, ], [expm, ])
        else:
            assert w.ndim == 1
            w = as_tensor_variable(w)
            return Apply(self, [A, w, ], [expm, ])

    def perform(self, node, inputs, outputs):
        (logm,) = outputs
        if self.mode == 'matrix': # default, as standard Theano Logm
            (A,) = inputs
            logm[0] = numpy.real(scipy.linalg.logm(A))
            #import scipy.linalg._matfuncs_inv_ssq
            #logm[0] = numpy.real(numpy.array(scipy.linalg._matfuncs_inv_ssq._logm(A))).astype(theano.config.floatX)
        elif self.mode == 'zeroest':
            (A,) = inputs
            hatxi = minimize(lambda hatxi: self.lossf(hatxi,A),
                    1e-6*numpy.random.rand(self.LAtoVf(A).shape[0]),
                    jac=lambda hatxi: self.dlossf(hatxi,A),
                    #method='COBYLA',
                    #constraints={'type':'ineq','fun':lambda hatxi: 1e-6-numpy.linalg.norm(self.expf(self.VtoLAf(hatxi))-A)**2},
                    ).x
            logm[0] = self.VtoLAf(hatxi)
        elif self.mode == 'nearest':
            (A,w,) = inputs
            hatxi = minimize(lambda hatxi: self.lossf(hatxi,A),
                    w,
                    jac=lambda hatxi: self.dlossf(hatxi,A),
                    ).x
            logm[0] = self.VtoLAf(hatxi)

    def infer_shape(self, node, shapes):
        return [shapes[0]]

    def grad(self, inputs, g_outputs):
        [gz] = g_outputs
        [A] = inputs
        v = self(A)

        dexp = T.jacobian(self.exp(v).flatten(),v)
        invdexp = T.nlinalg.matrix_inverse(dexp.reshape((A.shape[0]*A.shape[1],v.shape[0]*v.shape[1],))).reshape((A.shape[0],A.shape[1],v.shape[0],v.shape[1],))

        return [T.tensordot(gz,invdexp,((0,1),(0,1)))]

    def __str__(self):
        return "Logm"
