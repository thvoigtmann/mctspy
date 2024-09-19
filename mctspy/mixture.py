import numpy as np

from numba import njit
from .__util__ import model_base, void, nparray

def _dq (q):
    return np.diff(q, append=2*q[-1]-q[-2])

class mixture_model (model_base):
    """MCT model for simple-liquid mixtures.

    Parameters
    ----------
    Sq : structure_factor
        Should implement a structure factor; determines the number of
        species and concentrations used in the model.
    q : array_like
        Wave number grid, should follow the convention.
    D0 : float or array_like, default: 1.0
        The short-time diffusion coefficients; either an array whose
        length matches the number of species defined in the structure
        factor, or a scalar value that will be applied to all species.
    """
    def __init__ (self, Sq, q, D0=1.0):
        model_base.__init__(self)
        self.rho = Sq.density()
        self.q = q
        self.Sq = Sq
        self.sq, self.cq = Sq.Sq(q)
        self.M = q.shape[0]
        self.S = Sq.densities.shape[0]
        self.__init_vertices__ ()
        if isinstance(D0, (list, np.ndarray)):
          self.D0 = np.array(D0)
        else:
          self.D0 = np.ones(self.S)*D0
    def __len__ (self):
        return self.M
    def matrix_dimension (self):
        return self.S
    def scalar (self):
        return False
    def phi0 (self):
        return self.sq
    def Bq (self):
        return np.ones((self.M,self.S,self.S))*np.diag(1/self.D0)
    def Bqinv (self):
        return np.ones((self.M,self.S,self.S))*np.diag(self.D0)
    def Wq (self):
        return (self.q**2)[:,None,None] * np.linalg.inv(self.sq)
    def WqSq (self):
        return (self.q**2)[:,None,None] * np.diag(np.ones(self.S))
    def dq (self):
        return _dq(self.q)
    def __init_vertices__ (self):
        self.pre = 1./(32.*np.pi**2) * self.dq()[0]**2
        q = self.q
        self.a1 = np.zeros((3,self.M))
        self.a2 = np.zeros((3,self.M,self.S,self.S))
        self.a3 = np.zeros((3,self.M,self.S,self.S,self.S,self.S))
        self.b1 = np.zeros((3,self.M,self.M))
        self.b2 = np.zeros((3,self.M,self.M,self.S,self.S))
        self.b3 = np.zeros((3,self.M,self.M,self.S,self.S,self.S,self.S))
        self.a1[0] = q**5
        self.a1[1] = - 2 * q**3
        self.a1[2] =  q
        q = self.q[:,None,None]
        qcq = q * self.cq
        self.a2[0] = qcq
        self.a2[1] = - q**4 * qcq
        self.a2[2] = 2 * q**2 * qcq
        q = self.q[:,None,None,None,None]
        qcqcq = qcq[...,:,:,None,None]*self.cq[...,None,None,:,:]
        self.a3[0] = qcqcq * q**4
        self.a3[1] = qcqcq * 2 * q**2
        self.a3[2] = qcqcq
        q_, p_ = np.meshgrid(q,q,indexing='ij')
        pq = p_/q_**2
        self.b1[0] = pq
        self.b1[1] = pq * (q_**2 - p_**2)
        self.b1[2] = pq * (q_**2 - p_**2)**2
        q_, p_ = q_[...,None,None], p_[...,None,None]
        pcp = pq[:,:,None,None] * self.cq[None,:,:,:]
        self.b2[0] = (q_**4 - p_**4) * pcp
        self.b2[1] = pcp
        self.b2[2] = p_**2 * pcp
        q_, p_ = q_[...,None,None], p_[...,None,None]
        pcpcp = pcp[...,:,:,None,None]*self.cq[...,None,None,:,:]
        self.b3[0] = pcpcp
        self.b3[1] = (q_**2 + p_**2) * pcpcp
        self.b3[2] = (q_**2 + p_**2)**2 * pcpcp
        self.sqrtrho = np.sqrt(self.Sq.densities)
        #
        q = self.q
        for qi in range(self.M):
            assert(self.a1[0,qi] == q[qi]**5)
            assert(self.a1[1,qi] == - 2 * q[qi]**3)
            assert(self.a1[2,qi] == q[qi])
            for a in range(self.S):
                for b in range(self.S):
                    assert(self.a2[0,qi,a,b] == q[qi] * self.cq[qi,a,b])
                    assert(np.isclose(self.a2[1,qi,a,b] , - q[qi]**5 * self.cq[qi,a,b]))
                    assert(np.isclose(self.a2[2,qi,a,b] , 2*q[qi]**3 * self.cq[qi,a,b]))
                    for g in range(self.S):
                        for d in range(self.S):
                            assert(np.isclose(self.a3[0,qi,a,b,g,d] , q[qi]*self.cq[qi,a,b]*self.cq[qi,g,d] * q[qi]**4))
                            assert(np.isclose(self.a3[2,qi,a,b,g,d] , q[qi]*self.cq[qi,a,b]*self.cq[qi,g,d]))
            for pi in range(self.M):
                assert(self.b1[0,qi,pi] == q[pi]/q[qi]**2)
                for a in range(self.S):
                    for b in range(self.S):
                        assert(self.b2[1,qi,pi,a,b] == q[pi]/q[qi]**2 * self.cq[pi,a,b])
                        assert(np.isclose(self.b2[0,qi,pi,a,b],(q[qi]**4-q[pi]**4) * q[pi]/q[qi]**2 * self.cq[pi,a,b]))
                        for g in range(self.S):
                            for d in range(self.S):
                                assert(self.b3[0,qi,pi,a,b,g,d] == q[pi]/q[qi]**2 * self.cq[pi,a,b] * self.cq[pi,g,d])
                                assert(np.isclose(self.b3[1,qi,pi,a,b,g,d] , (q[qi]**2 + q[pi]**2) * q[pi]/q[qi]**2 * self.cq[pi,a,b] * self.cq[pi,g,d]))
    def make_kernel (self):
        a1, a2, a3 = self.a1, self.a2, self.a3
        b1, b2, b3 = self.b1, self.b2, self.b3
        M, S = self.M, self.S
        sr = self.sqrtrho
        q = self.q
        dq = self.dq()
        pre = self.pre
        @njit
        def ker (m, phi, i, t):
            for qi in range(M):
                for a in range(S):
                    for b in range(S):
                        mq = 0.
                        for n in range(3):
                            pi, ki = qi, 0
                            zqk1 = b1[n,qi,pi] * phi[pi,a,b]
                            zqk2 = 0.0
                            for g in range(S):
                                zqk2 += b2[n,qi,pi,b,g] *  phi[pi,a,g] * sr[g]
                            for g in range(S):
                                mq += a2[n,ki,a,g] * zqk2 * phi[ki,b,g] * sr[g]
                                for d in range(S):
                                    mq += a3[n,ki,a,g,b,d] * zqk1 \
                                          * phi[ki,g,d] * sr[g] * sr[d]
                            for ki in range(1, M):
                                if ki <= qi:
                                    pi = qi - ki
                                    sgn = 1
                                else:
                                    pi = ki - qi - 1
                                    sgn = -1
                                zqk1 += sgn*b1[n,qi,pi] * phi[pi,a,b]
                                for g in range(S):
                                    zqk2 += sgn*b2[n,qi,pi,b,g]*phi[pi,a,g]*sr[g]
                                pi = qi + ki
                                if pi < M:
                                    zqk1 += b1[n,qi,pi] * phi[pi,a,b]
                                    for g in range(S):
                                        zqk2 += b2[n,qi,pi,b,g]*phi[pi,a,g]*sr[g]
                                for g in range(S):
                                    mq += a2[n,ki,a,g]*zqk2*phi[ki,b,g]*sr[g]
                                    for d in range(S):
                                        mq += a3[n,ki,a,g,b,d] * zqk1 \
                                              * phi[ki,g,d] * sr[g] * sr[d]
                        mq *= 2.0
                        m[qi,a,b] = mq * pre / q[qi]
        return ker
