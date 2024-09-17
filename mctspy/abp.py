import numpy as np
import scipy.linalg as la

from numba import njit
from .__util__ import model_base, void, nparray

def _dq (q):
    return np.diff(q, append=2*q[-1]-q[-2])

class abp_model_2d (model_base):
    def __init__ (self, Sq, q, L=1, D0=1.0, Dr=1.0, v0=0.0):
        model_base.__init__(self)
        self.dtype = np.dtype('complex')
        self.rho = Sq.density()
        self.q = q
        self.Sq = Sq
        self.sq, self.cq = Sq.Sq(q)
        self.M = q.shape[0]
        self.L = L
        self.S = 2*L+1
        self.D0 = D0
        self.Dr = Dr
        self.v0 = v0
        self.__init_vertices__ ()
    def __len__ (self):
        return self.M
    def matrix_dimension (self):
        return self.S
    def scalar (self):
        return False
    def phi0 (self):
        phi0 = np.ones_like(self.q)[:,None,None] * np.diag(np.ones(self.S))
        phi0[:,self.L,self.L] = self.sq
        return phi0
    def Bq (self):
        return self.wTinv
    def Bqinv (self):
        return self.omega_T(self.L)
    def Wq (self):
        res = self.WqSq()
        res[:,self.L,self.L] = res[:,self.L,self.L] * self.sq
        return res
    def WqSq (self):
        wTinv_wR = np.zeros((self.M,self.S,self.S),dtype=self.dtype)
        for q in range(self.M):
            wTinv_wR[q] = np.dot(self.wTinv[q],
                          self.Dr * np.diag(np.arange(-self.L, self.L+1)**2))
        return (self.q**2)[:,None,None] * np.diag(np.ones(self.S)) \
               + wTinv_wR
    def dq (self):
        return _dq(self.q)
    def omega_T (self, Lcut):
        L, S = Lcut, 2*Lcut+1
        wT = np.zeros((self.M,S,S),dtype=self.dtype)
        for l in range(-L,L+1):
            for ld in range(-L,L+1):
                if l==ld:
                    wT[:,L+l,L+ld] = self.D0
                if abs(l-ld)==1:
                    if l==0:
                        wT[:,L+l,L+ld] = - 0.5j/self.q*self.v0 * self.sq
                    else:
                        wT[:,L+l,L+ld] = - 0.5j/self.q*self.v0
        return wT
    def omega_R (self, Lcut):
        L, S = Lcut, 2*Lcut+1
        return np.ones((self.M,S,S),dtype=self.dtype) \
               * self.Dr *np.diag(np.arange(-Lcut, Lcut+1)**2)
    def low_density_solution (self, t, Lcut):
        L, S = Lcut, 2*Lcut+1
        phi = np.zeros((t.shape[0], self.M, S, S), dtype=self.dtype)
        w = (self.q**2)[:,None,None] * self.omega_T(Lcut) + self.omega_R(Lcut)
        for q in range(self.M):
            wqt = np.einsum('i,jk->ijk', t, -w[q])
            phi[:,q,:,:] = la.expm(wqt)
        return phi
    def __init_vertices__ (self):
        self.pre = 1./(32.*np.pi**2) * self.dq()[0]**2
        q = self.q
        L=self.L
        Delta = np.sqrt(self.D0**2 + (self.v0/self.q)**2)
        wTinv0 = np.zeros((self.M,self.S,self.S),dtype=self.dtype)
        for l in range(-L,L+1):
            for ld in range(-L,L+1):
                wTinv0[:,L+l,L+ld] = np.power(1j*self.v0/self.q/ \
                                     (self.D0 + Delta),abs(l-ld)) / Delta
        self.wTinv = wTinv0.copy()
        if L > 0:
            u0 = -0.5j*self.q*self.v0/self.D0 * (self.sq - 1)
            print (u0)
            for l in range(-L,L+1):
                for ld in range(-L,L+1):
                    self.wTinv[:,L+l,L+ld] -= u0*wTinv0[:,L+l,L] * \
                        (wTinv0[:,L+1,L+ld]+wTinv0[:,L-1,L+ld])/ \
                        (self.q**2 + u0*(wTinv0[:,L+1,L]+wTinv0[:,L-1,L]))
    def make_kernel (self):
        M, S = self.M, self.S
        @njit
        def ker (m, phi, i, t):
            for qi in range(M):
                for a in range(S):
                    for b in range(S):
                        m[qi,a,b] = 0.+1j*0 # mq * pre / q[qi]
        return ker
