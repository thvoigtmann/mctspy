import numpy as np

from numba import njit
from .__util__ import model_base, void, nparray

def _dq (q):
    return np.diff(q, append=2*q[-1]-q[-2])

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
    def dq (self):
        return _dq(self.q)
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
        dq = self.dq()
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
    def set_C (self, f):
        a, b, apre = self.a, self.b, self.apre
        M = self.M
        C = np.zeros((M,M))
        @njit
        def calc_C (C, f):
            for qi in range(M):
                for n in range(9):
                    pi, ki = qi, 0
                    zqk = b[n,qi,qi] * f[pi]
                    C[qi,ki] += a[n,0] * zqk * (1-f[ki])**2
                    for ki in range(1, M):
                        if ki <= qi:
                            pi = qi - ki
                            zqk += b[n,qi,pi] * f[pi]
                        else:
                            pi = ki - qi - 1
                            zqk -= b[n,qi,pi] * f[pi]
                        pi = qi + ki
                        if pi < M:
                            zqk += b[n,qi,pi] * f[pi]
                        C[qi,ki] += a[n,ki] * zqk * (1-f[ki])**2
        calc_C (C, f)
        self.__C__ = C
    def make_dm (self, m, phi, dphi):
        M = self.M
        Cqk = void(self.__C__)
        q = self.q
        dq = self.dq()
        @njit
        def dm(m, phi, dphi):
            V = nparray(Cqk)
            m[:] = 2 * dq**2/q**2 * np.dot(V,dphi)
        return dm
    def make_dmhat (self, m, f, ehat):
        M = self.M
        Cqk = void(self.__C__)
        q = self.q
        dq = self.dq()
        @njit
        def dmhat(m, f, ehat):
            V = nparray(Cqk)
            m[:] = np.dot(ehat/q**2, V) * 2 * dq**2
        return dmhat
    def make_dm2 (self, m, phi, dphi):
        M = self.M
        a, b, apre = self.a, self.b, self.apre
        q = self.q
        dq = self.dq()
        def dm2 (m, phi, dphi):
            for qi in range(M):
                mq = 0.
                for n in range(6):
                    pi, ki = qi, 0
                    zqk = b[n,qi,qi] * dphi[pi] * (1-phi[pi])**2
                    mq += a[n,0] * apre[n] * zqk * dphi[ki] * (1-phi[ki])**2
                    for ki in range(1, M):
                        if ki <= qi:
                            pi = qi - ki
                            zqk += b[n,qi,pi] * dphi[pi] * (1-phi[pi])**2
                        else:
                            pi = ki - qi - 1
                            zqk -= b[n,qi,pi] * dphi[pi] * (1-phi[pi])**2
                        pi = qi + ki
                        if pi < M:
                            zqk += b[n,qi,pi] * dphi[pi] * (1-phi[pi])**2
                        mq += a[n,ki] *apre[n]* zqk * dphi[ki] * (1-phi[ki])**2
                m[qi] = mq * dq[qi]**2 / q[qi]**2
        return dm2

    def h5save (self, fh):
        grp = fh.create_group("model")
        grp.attrs['type'] = 'simple_liquid'
        grp.attrs['M'] = self.M
        grp.attrs['dynamics'] = 'BD'
        grp.attrs['D0'] = self.D0
        grp.attrs['rho'] = self.rho
        grp.create_dataset("q",data=self.q)
        grp.create_dataset("sq",data=self.sq)
        grp.create_dataset("cq",data=self.cq)

class tagged_particle_model (model_base):
    def __init__ (self, base_model, cs, D0s=1.0):
        self.base = base_model
        self.q = base_model.q
        self.cs = cs.cq(self.q)
        self.M = self.q.shape[0]
        self.__init_vertices__()
        self.D0 = D0s
    def __len__ (self):
        return self.M
    def Wq (self):
        return self.q*self.q
    def Bq (self):
        return np.ones(self.M)/self.D0
    def __init_vertices__ (self):
        pre = (1./(4.*np.pi))**2. * self.base.rho
        print("TP",pre)
        q = self.q
        sp = self.base.sq
        csp = self.cs
        self.a = np.zeros((3,self.M))
        self.b = np.zeros((3,self.M,self.M))
        self.a[0] = q**4 * q
        self.a[1] = - 2 * q**2 * q
        self.a[2] = q
        q_, p_ = np.meshgrid(q,q,indexing='ij')
        pq = p_/q_**3 * pre * sp * csp**2
        qsq_psq = q_**2 - p_**2
        self.b[0] = pq
        self.b[1] = pq * qsq_psq
        self.b[2] = pq * qsq_psq**2
    def set_base (self, array):
        model_base.set_base (self, array)
        a, b = self.a, self.b
        V = np.zeros((self.M,self.M))
        M = self.M
        @njit
        def calc_V (V, f):
            for qi in range(M):
                for ki in range(M):
                    V[qi,ki] = 0.
                for n in range(3):
                    pi, ki = qi, 0
                    zqk = b[n,qi,pi] * f[pi]
                    V[qi,ki] += a[n,ki] * zqk
                    for ki in range(1, M):
                        if ki <= qi:
                            pi = qi - ki
                            zqk += b[n,qi,pi] * f[pi]
                        else:
                            pi = ki - qi - 1
                            zqk -= b[n,qi,pi] * f[pi]
                        pi = qi + ki
                        if pi < M:
                            zqk += b[n,qi,pi] * f[pi]
                        V[qi,ki] += a[n,ki] * zqk
        #calc_V (V, array[0])
        self.__calcV__ = calc_V
        self.__Vqk__ = V
    def make_kernel (self, ms, phis, i, t):
        M = self.M
        Vqk = void(self.__Vqk__)
        Vfunc = self.__calcV__
        base_phi = self.base.phi
        dq = _dq(self.base.q)
        self.__i__ = np.zeros(1,dtype=int)
        __i__ = void(self.__i__)
        @njit
        def ker (ms, phis, i, t):
            last_i = nparray(__i__)
            V = nparray(Vqk)
            phi = nparray(base_phi)
            if last_i[0]==0 or not (i==last_i[0]):
                Vfunc(V, phi[i])
                last_i[0] = i
            ms[:] = dq**2 * np.dot(V,phis)
        return ker


class tagged_particle_q0 (model_base):
    def __init__ (self, base_model):
        self.base = base_model
        self.__init_vertices__()
    def __len__ (self):
        return 1
    def __init_vertices__ (self):
        pre = 1./(6.*np.pi**2) * _dq(self.base.base.q)[0] * self.base.base.rho
        sk = self.base.base.sq
        cs = self.base.cs
        self.V = pre * (self.base.q**2 * cs)**2 * sk
    def set_base (self, array):
        model_base.set_base (self, array)
    def make_kernel (self, ms0, phis0, i, t):
        M = self.base.M
        Vk = void(self.V)
        base_phi_s = self.base.phi
        base_phi = self.base.base.phi
        @njit
        def ker (ms0, phis0, i, t):
            phi = nparray(base_phi)
            phi_s = nparray(base_phi_s)
            V = nparray(Vk)
            ms0[:] = np.dot(V,phi[i]*phi_s[i])
        return ker
