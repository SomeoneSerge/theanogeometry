# # This file is part of Theano Geometry
#
# Copyright (C) 2017, Stefan Sommer (sommer@di.ku.dk)
# https://bitbucket.org/stefansommer/theanogemetry
#
# Theano Geometry is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Theano Geometry is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Theano Geometry. If not, see <http://www.gnu.org/licenses/>.
#

from src.setup import *
from src.utils import *

def initialize(M):
    """ Riemannian curvature tensor """

    d = M.dim
    x = M.element()
    nu = M.frame()
    e1 = M.vector()
    e2 = M.vector()
    
    """
    Riemannian Curvature tensor

    Args:
        x: point on manifold

    Returns:
        4-tensor R_ijk^m in with order i,j,k,m
    """
    def R(x):
        return (
                T.tensordot(M.Gamma_g(x),M.Gamma_g(x),(0,2)).dimshuffle(3,0,1,2) 
                - T.tensordot(M.Gamma_g(x),M.Gamma_g(x),(0,2)).dimshuffle(0,3,1,2) 
                +  T.jacobian(M.Gamma_g(x).flatten(),x).reshape((d,d,d,d)).dimshuffle(3,1,2,0)
                -  T.jacobian(M.Gamma_g(x).flatten(),x).reshape((d,d,d,d)).dimshuffle(1,3,2,0) 
                )
    M.R = R
    M.Rf = theano.function([x], R(x))

    """
    Riemannian Curvature form
    R_u (also denoted Omega) is the gl(n)-valued curvature form u^{-1}Ru for a frame
    u for T_xM

    Args:
        x: point on manifold

    Returns:
        4-tensor (R_u)_ijk^m in with order i,j,k,m
    """
    def R_u(x,u):
        return T.tensordot(T.nlinalg.matrix_inverse(u),T.tensordot(R(x),u,(2,0)),(1,2)).dimshuffle(1,2,3,0)
    M.R_u = R_u
    M.R_uf = theano.function([x,nu], R_u(x,nu))

    """
    Sectional curvature

    Args:
        x: point on manifold
        e1,e2: two orthonormal vectors spanning the section

    Returns:
        sectional curvature K(e1,e2)
    """
    def sec_curv(x,e1,e2):
        Rflat = T.tensordot(M.R(x),M.g(x),[3,0])
        sec = T.tensordot(
                T.tensordot(
                    T.tensordot(
                        T.tensordot(
                            Rflat, 
                            e1, [0,0]), 
                        e2, [0,0]),
                    e2, [0,0]), 
                e1, [0,0])
        return sec
    M.sec_curv = sec_curv
    M.sec_curvf = theano.function([x,e1,e2],sec_curv(x,e1,e2))

    """
    Ricci curvature

    Args:
        x: point on manifold

    Returns:
        2-tensor R_ij in order i,j
    """
    Ricci_curv = lambda x: T.tensordot(M.R(x),T.eye(M.dim),((0,3),(0,1)))
    M.Ricci_curv = Ricci_curv
    M.Ricci_curvf = theano.function([x],Ricci_curv(x))

    """
    Scalar curvature

    Args:
        x: point on manifold

    Returns:
        scalar curvature
    """
    S_curv = lambda x: T.tensordot(M.Ricci_curv(x),M.gsharp(x),((0,1),(0,1)))
    M.S_curv = S_curv
    M.S_curvf = theano.function([x],S_curv(x))
