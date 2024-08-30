import numpy as np

from numba import njit
from .__util__ import model_base

class simple_liquid_model (model_base):
    def __init__ (self, Sq, q, D0=1.0):
        self.rho = Sq.density()
        self.q = q
        self.sq, self.cq = Sq.Sq(q)
        self.M = q.shape[0]
        self.__init_vertices__()
        self.D0 = D0
    def __len__ (self):
        return self.M
    def Wq (self):
        return self.q*self.q
    def Bq (self):
        return self.sq/self.D0
    def __init_vertices__ (self):
        pre = 1./(32.*np.pi**2) * self.rho
        q = self.q
        qSq = q * self.sq
        self.a = np.zeros((9,self.M))
        self.b = np.zeros((9,self.M,self.M))
        self.a[0] = qSq * 2 * self.cq
        self.a[1] = - qSq * 2 * q**4 * self.cq
        self.a[2] = qSq * 4 * q**2 * self.cq
        self.a[3] = qSq * q**4 * (self.cq)**2
        self.a[4] = qSq * 2 * q**2 * (self.cq)**2
        self.a[5] = qSq * (self.cq)**2
        self.a[6] = qSq * q**4
        self.a[7] = - qSq * 2 * q**2
        self.a[8] = qSq
        self.apre = np.array([1,1,1,2,2,2,2,2,2])
        q_, p_ = np.meshgrid(q,q,indexing='ij')
        pSp = p_/q_ * np.outer(self.sq, self.sq) * pre / q_**2
        cq, cp = np.meshgrid(self.cq, self.cq, indexing='ij')
        qsq_psq = q_**2 - p_**2
        qsqppsq = q_**2 + p_**2
        self.b[0] = pSp * (q_**4 - p_**4) * cp
        self.b[1] = pSp * cp
        self.b[2] = pSp * p_**2 * cp
        self.b[3] = pSp
        self.b[4] = pSp * qsq_psq
        self.b[5] = pSp * qsq_psq**2
        self.b[6] = pSp * cp**2
        self.b[7] = pSp * qsqppsq * cp**2
        self.b[8] = pSp * qsqppsq**2 * cp**2
    def make_kernel (self, m, phi, i, t):
        a, b, apre = self.a, self.b, self.apre
        M = self.M
        dq = np.diff(self.q, append=2*self.q[-1]-self.q[-2])
        @njit
        def ker (m, phi, i, t):
            for qi in range(M):
                mq = 0.
                for n in range(6):
                    pi, ki = qi, 0
                    zqk = b[n,qi,qi] * phi[pi]
                    mq += a[n,0] * apre[n] * zqk * phi[ki]
                    for ki in range(1, M):
                        if ki <= qi:
                            pi = qi - ki
                            zqk += b[n,qi,pi] * phi[pi]
                        else:
                            pi = ki - qi - 1
                            zqk -= b[n,qi,pi] * phi[pi]
                        pi = qi + ki
                        if pi < M:
                            zqk += b[n,qi,pi] * phi[pi]
                        mq += a[n,ki] * apre[n] * zqk * phi[ki]
                m[qi] = mq * dq[qi]**2
        return ker
