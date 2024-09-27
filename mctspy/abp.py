import numpy as np
import scipy.linalg as la

from numba import njit, complex128
from .__util__ import model_base, void, nparray
from .standard_2d import g0unsum, g1unsum, g2unsum

def _dq (q):
    return np.diff(q, append=2*q[-1]-q[-2])

@njit
def _calc_phase(phase,L,q,k,p,x,y):
    for l1 in range(-L,L+1):
        for l2 in range(-L,L+1):
            for l3 in range(-L,L+1):
                for l4 in range(-L,L+1):
                    mu, nu = l3-l1, l4-l2
                    smu, snu = np.sign(mu), np.sign(nu)
                    a = np.arange(abs(mu)+1)[:,None]
                    b = np.arange(abs(nu)+1)[None,:]
                    phase[:,L+l1,L+l2,L+l3,L+l4] = np.sum(np.sum(x[:,None,None]**(abs(mu)-a) * y[:,None,None]**a * (smu*1j)**a * (-snu*1j)**b * (k/p)[:,None,None]**abs(nu) * (q/k-x)[:,None,None]**(abs(nu)-b) * y[:,None,None]**b, axis=-1), axis=-1)

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
        res[:,self.L,self.L] = res[:,self.L,self.L] / self.sq
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
            for l in range(-L,L+1):
                for ld in range(-L,L+1):
                    self.wTinv[:,L+l,L+ld] -= u0*wTinv0[:,L+l,L] * \
                        (wTinv0[:,L+1,L+ld]+wTinv0[:,L-1,L+ld])/ \
                        (self.q**2 + u0*(wTinv0[:,L+1,L]+wTinv0[:,L-1,L]))
        self.__A__ = np.zeros((self.M,self.M,self.M),dtype=self.dtype)
        self.__B__ = np.zeros((self.M,self.M,self.M,self.S,self.S),dtype=self.dtype)
        self.__C__ = np.zeros((self.M,self.M,self.M,self.S,self.S),dtype=self.dtype)
        self.__D__ = np.zeros((self.M,self.M,self.M,self.S,self.S),dtype=self.dtype)
        #self.__phase__ = np.zeros((self.M,self.S,self.S,self.S,self.S),dtype=self.dtype)
        A, B, C, D = self.__A__, self.__B__, self.__C__, self.__D__
        q = self.q
        M = self.M
        lr = np.arange(-L,L+1)
        pre = self.rho/(8*np.pi**2)
        Pe_t = self.v0/self.D0
        v0pre = 0.5j*Pe_t*self.sq
        c = self.cq
        for qi in range(M):
            for ki in range(M):
                if ki <= qi:
                    pmin = q[qi] - q[ki]
                else:
                    pmin = q[ki] - q[qi]
                pmax = q[qi] + q[ki]
                x = (q[qi]**2 + q[ki]**2 - q**2) / (2*q[qi]*q[ki])
                mask = (x>=-1) & (x<=1) & (q>=pmin)
                x = x[mask]
                p = np.arange(M)[mask]
                minval = -1.0
                if pmax > q[-1]:
                    minval = x[-1]
                y = np.sqrt(1-x*x)
                thk = np.power(x[:,None]+1j*y[:,None],lr)
                thp = np.power(q[ki]/x[:,None],lr) * np.power((q[qi]/q[ki]-x[:,None])-1j*y[:,None],lr)
                assert (not np.isnan(thk).any())
                assert (not np.isnan(thp).any())
                for l in lr:
                    for ld in lr:
                        A[qi,ki,p] = pre * (2*q[qi]**2 * g0unsum(c[p]**2,x,1,minval) \
                                 - 4*q[qi]*q[ki] * g1unsum(c[p]**2,x,1,minval) \
                                 + 2*q[ki]**2 * g2unsum(c[p]**2,x,1,minval) )
                        B[qi,ki,p,L+l,L+ld] = pre * (2*q[qi]*q[ki] * g1unsum(thp[:,L+ld]*thk[:,L-l]*c[ki]*c[p],x,1,minval) \
                                 - 2*q[ki]**2 * g2unsum(thp[:,L+ld]*thk[:,L-l]*c[ki]*c[p],x,1,minval) )
                        C[qi,ki,p,L+l,L+ld] = pre * (2*q[qi]*q[ki] * g1unsum(thp[:,L-l]*thk[:,L+ld]*c[ki]*c[p],x,1,minval) \
                                 - 2*q[ki]**2 * g2unsum(thp[:,L-l]*thk[:,L+ld]*c[ki]*c[p],x,1,minval) )
                        D[qi,ki,p,L+l,L+ld] = pre * 2*q[ki]**2 * g2unsum(thp[:,L+ld]*thp[:,L-l]*c[ki]**2,x,1,minval)
                        if not Pe_t==0:
                            if L>=1:
                                A[qi,ki,L+l,L+ld] += v0pre[qi] * ( \
                                    2 * g0((phi[p,L-1,L]+phi[p,L+1,L])*p*Sq[p]*c[p]**2*thk[:,L-l]*thk[:,L+ld],x,1,minval) \
                                    - 2*q[ki]/q[qi] * g1((phi[p,L-1,L]+phi[p,L+1,L])*p*Sq[p]*c[p]**2*thk[:,L-l]*thk[:,L+ld],x,1,minval) )
                                B[qi,ki,L+l,L+ld] += v0pre[qi] * \
                                    2*q[ki]/q[qi] * g1((phi[p,L-1,L+ld]+phi[p,L+1,L+ld])*p*Sq[p]*c[p]*c[ki]*thk[:,L-l]*thp[:,L+ld],x,1,minval)
                            if L>=abs(l)+1:
                                C[qi,ki,L+l,L+ld] += v0pre[qi] * ( \
                                    - 2 * g0((phi[p,L+l-1,L]*thk[:,L-1]*thp[:,L+1]+phi[p,L+l+1,L]*thk[:,L+1]*thp[:,L-1])*q[ki]*c[ki]*c[p]*thk[:,L+ld]*thp[:,L-l],x,1,minval) \
                                    + 2*q[ki]/q[qi] * g1((phi[p,L+l-1,L]*thk[:,L-1]*thp[:,L+1]+phi[p,L+l+1,L]*thk[:,L+1]*thp[:,L-1])*q[ki]*c[ki]*c[p]*thk[:,L+ld]*thp[:,L-l],x,1,minval) )
                                D[qi,ki,L+l,L+ld] += v0pre[qi] * \
                                    (-2) * q[ki]/q[qi] * g1((phi[p,L+l-1,L+ld]*thk[:,L-1]*thp[:,L+1]+phi[p,L+l+1,L+ld]*thk[:,L+1]*thp[:,L-1])*q[ki]*c[ki]**2*thp[:,L-l]*thp[:,L+ld],x,1,minval)
                assert (not np.isnan(A[qi,ki]).any())
                assert (not np.isnan(B[qi,ki]).any())
                assert (not np.isnan(C[qi,ki]).any())
                assert (not np.isnan(D[qi,ki]).any())

    def make_kernel (self):
        M, S, L = self.M, self.S, self.L
        q = self.q
        pre = self.rho/(8*np.pi**2)
        Aqk = void(self.__A__)
        Bqk = void(self.__B__)
        Cqk = void(self.__C__)
        Dqk = void(self.__D__)
        #Pqk = void(self.__phase__)
        omega_T_inv = void(self.wTinv)
        Sq, c = self.sq, self.cq
        Pe_t = self.v0/self.D0
        @njit
        def ker (m, phi, i, t):
            A = nparray(Aqk)
            B = nparray(Bqk)
            C = nparray(Cqk)
            D = nparray(Dqk)
            #phase = nparray(Pqk)
            wTinv = nparray(omega_T_inv)
            lr = np.arange(-L,L+1)
            Ainner = np.sum(A*phi[:,L,L],axis=-1)
            Binner = np.sum(B*phi[:,L,None,:],axis=-3)
            Cinner = np.sum(C*phi[:,:,L,None],axis=-3)
            Dinner = np.sum(D*phi[:,:,:],axis=-3)
            m[:,:,:] = - ( \
                np.sum((q[:-1,None,None]*phi[:-1,:,:]*Ainner[:,:-1,None,None] + q[1:,None,None]*phi[1:,:,:]*Ainner[:,1:,None,None])/2 * np.diff(q)[:,None,None], axis=1) \
                + np.sum((q[:-1,None,None]*phi[:-1,:,L,None]*Binner[:,:-1,:,:] + q[1:,None,None]*phi[1:,:,L,None]*Binner[:,1:,:,:])/2 * np.diff(q)[:,None,None], axis=1) \
                + np.sum((q[:-1,None,None]*phi[:-1,L,None,:]*Cinner[:,:-1,:,:] + q[1:,None,None]*phi[1:,L,None,:]*Cinner[:,1:,:,:])/2 * np.diff(q)[:,None,None], axis=1) \
                + np.sum((q[:-1,None,None]*phi[:-1,L,L,None,None]*Dinner[:,:-1,:,:] + q[1:,None,None]*phi[1:,L,L,None,None]*Dinner[:,1:,:,:])/2 * np.diff(q)[:,None,None], axis=1))
            for qi in range(M):
                m[qi,:,:] = np.dot(wTinv[qi],np.dot(m[qi],wTinv[qi]))
        return ker
