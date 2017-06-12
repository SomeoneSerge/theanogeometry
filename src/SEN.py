from src.params import *
from src.group import *

##########################################################################
# this file contains definitions for G=SO(3)                             #
##########################################################################

DIM = N*(N-1)//2 # group dimension
MDIM = N*N # matrix/embedding space dimension

## coordinate chart linking Lie algebra LA={A\in\RR^{NxN}|\trace{A}=0} and V=\RR^DIM
# derived from https://stackoverflow.com/questions/25326462/initializing-a-symmetric-theano-dmatrix-from-its-upper-triangle
r = T.arange(N)
tmp_mat = r[np.newaxis, :] + ((N * (N - 3)) // 2-(r * (r - 1)) // 2)[::-1,np.newaxis]
tmp_mat1 = T.triu(tmp_mat+1)-T.diag(T.diagonal(tmp_mat+1))
triu_index_matrix = tmp_mat1 + tmp_mat1.T

VtoLA = lambda v: T.concatenate((T.zeros(1),v))[triu_index_matrix] # from \RR^DIM to LA
LAtoV = lambda m: m[T.triu(T.ones((N,N))-T.diag(T.ones(N))).nonzero()] # from LA to \RR^DIM

triu_index_matrixf = theano.function([], triu_index_matrix)
VtoLAf = theano.function([v], VtoLA(v))
LAtoVf = theano.function([m], LAtoV(m),on_unused_input='ignore')

# print(triu_index_matrixf())
# print(VtoLAf(np.arange(1., 16.).astype(np.float32)))
# print(LAtoVf(VtoLAf(np.arange(1., 16.).astype(np.float32))))
