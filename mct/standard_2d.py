import numpy as np

from numba import njit
from .util import filon_integrate
from .__util__ import model_base, void, nparray

@njit
def g0(f,x):
    G0 = np.arcsin(x)
    G1 = -np.sqrt(1-x*x)
    b = np.diff(f)/np.diff(x) # axis=-1 is default here
    a = f[...,:-1] - x[...,:-1]*b
    return np.sum(a*np.diff(G0) + b*np.diff(G1),axis=-1)
@njit
def g1(f,x):
    G1 = -np.sqrt(1-x*x)
    G2 = 0.5*np.arcsin(x) - 0.5*x*np.sqrt(1-x*x)
    b = np.diff(f)/np.diff(x) # axis=-1 is default here
    a = f[...,:-1] - x[...,:-1]*b
    return np.sum(a*np.diff(G1) + b*np.diff(G2),axis=-1)
@njit
def g2(f,x):
    G2 = 0.5*np.arcsin(x) - 0.5*x*np.sqrt(1-x*x)
    G3 = (-1./3)*np.sqrt(1-x*x) * (2+x*x)
    b = np.diff(f)/np.diff(x) # axis=-1 is default here
    a = f[...,:-1] - x[...,:-1]*b
    return np.sum(a*np.diff(G2) + b*np.diff(G3),axis=-1)
    


# think of making this a subclass of simple_liquid_model?
class simple_liquid_model_2d (model_base):
    def __init__ (self, Sq, q, D0=1.0):
        self.rho = Sq.density()
        self.q = q
        self.sq, self.cq = Sq.Sq(q)
        self.M = q.shape[0]
        self.__init_vertices__ ()
        self.D0 = D0
    def __len__ (self):
        return self.M
    def Wq (self):
        return self.q*self.q
    def Bq (self):
        return self.sq/self.D0
    def __init_vertices__ (self):
        self.__A__ = np.zeros((self.M,self.M))
    def make_kernel (self, m, phi, i, t):
        q = self.q
        M = self.M
        pre = self.rho*self.sq/(8*np.pi**2*q**2)
        Aqk = void(self.__A__)
        c = self.cq
        S = self.sq
        @njit
        def ker (m, phi, i, t):
            A = nparray(Aqk)
            for qi in range(M):
                for ki in range(M):
                    if ki <= qi:
                        pmin = qi - ki
                    else:
                        pmin = ki - qi
                    pmax = qi + ki + 1
                    if pmax > M:
                        pmax = M
                    assert(pmax >= pmin)
                    p = np.arange(pmin,pmax)
                    x = (q[qi]**2 + q[ki]**2 - q[p]**2) / (2*q[qi]*q[ki])
                    assert((x>-1).all())
                    assert((x<1).all())
                    A[qi,ki] = 2*q[qi]**4 * g0(phi[p]*S[p]*c[p]**2,x) \
                          + 4*q[qi]**3*q[ki] * g1(phi[p]*S[p]*c[p]*(c[ki]-c[p]),x) \
                          + 2*q[qi]**2*q[ki]**2 * g2(phi[p]*S[p]*(c[ki]-c[p])**2,x)
                    #assert(not np.isnan(A[qi,ki]))
            m[:] = -pre * np.sum((q[:-1]*phi[:-1]*S[:-1]*A[:,:-1] + q[1:]*phi[1:]*S[1:]*A[:,1:])/2 * np.diff(q), axis=-1)
        return ker
