import numpy as np

from numba import njit
from .util import filon_integrate
from .__util__ import model_base, void, nparray

# try to perform the integrals well, going correctly to the boundaries
# with extrapolation if possible; the interval is [-1,1] usually,
# but the kernel method may adjust the -1 in order to stop wrong
# extrapolations occurring near the cutoff wave vector
# as it stands, the method now seems to be similar to what Caraglio et al
# show, perhaps even slightly better in terms of convergence with M

@njit
def g0(f,x,x0,x1):
    G0 = np.arcsin(x)
    G1 = -np.sqrt(1-x*x)
    G00,G01 = np.arcsin(x0),np.arcsin(x1)
    G10,G11 = -np.sqrt(1-x0*x0),-np.sqrt(1-x1*x1)
    b = np.diff(f)/np.diff(x) # axis=-1 is default here
    a = f[...,:-1] - x[...,:-1]*b
    if len(x)>1:
        bd1 = a[0]*(G0[0]-G00) + b[0]*(G1[0]-G10)
        bd2 = a[-1]*(G01-G0[-1]) + b[-1]*(G11-G1[-1])
    else:
        bd1 = f[0]*(G01-G00)
        bd2 = 0
    return np.sum(a*np.diff(G0) + b*np.diff(G1),axis=-1) + bd1 + bd2
@njit
def g1(f,x,x0,x1):
    G1 = -np.sqrt(1-x*x)
    G2 = 0.5*np.arcsin(x) - 0.5*x*np.sqrt(1-x*x)
    G10,G11 = -np.sqrt(1-x0*x0),-np.sqrt(1-x1*x1)
    G20,G21 = 0.5*np.arcsin(x0) - 0.5*x0*np.sqrt(1-x0*x0), \
              0.5*np.arcsin(x1) - 0.5*x1*np.sqrt(1-x1*x1)
    b = np.diff(f)/np.diff(x) # axis=-1 is default here
    a = f[...,:-1] - x[...,:-1]*b
    if len(x)>1:
        bd1 = a[0]*(G1[0]-G10) + b[0]*(G2[0]-G20)
        bd2 = a[-1]*(G11-G1[-1]) + b[-1]*(G21-G2[-1])
    else:
        bd1 = f[0]*(G11-G10)
        bd2 = 0
    return np.sum(a*np.diff(G1) + b*np.diff(G2),axis=-1) + bd1 + bd2
@njit
def g2(f,x,x0,x1):
    G2 = 0.5*np.arcsin(x) - 0.5*x*np.sqrt(1-x*x)
    G3 = (-1./3)*np.sqrt(1-x*x) * (2+x*x)
    G20,G21 = 0.5*np.arcsin(x0) - 0.5*x0*np.sqrt(1-x0*x0), \
              0.5*np.arcsin(x1) - 0.5*x1*np.sqrt(1-x1*x1)
    G30,G31 = (-1./3)*np.sqrt(1-x0*x0) * (2+x0*x0), \
              (-1./3)*np.sqrt(1-x1*x1) * (2+x1*x1)
    b = np.diff(f)/np.diff(x) # axis=-1 is default here
    a = f[...,:-1] - x[...,:-1]*b
    if len(x)>1:
        bd1 = a[0]*(G2[0]-G20) + b[0]*(G3[0]-G30)
        bd2 = a[-1]*(G21-G2[-1]) + b[-1]*(G31-G3[-1])
    else:
        bd1 = f[0]*(G21-G20)
        bd2 = 0
    return np.sum(a*np.diff(G2) + b*np.diff(G3),axis=-1) + bd1 + bd2



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
        if False:
            M = self.M
            q = self.q
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
                    print(qi,ki,x)
            quit()
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
                    A[qi,ki] = 2*q[qi]**4 * g0(phi[p]*S[p]*c[p]**2,x,1,minval) \
                          + 4*q[qi]**3*q[ki] * g1(phi[p]*S[p]*c[p]*(c[ki]-c[p]),x,1,minval) \
                          + 2*q[qi]**2*q[ki]**2 * g2(phi[p]*S[p]*(c[ki]-c[p])**2,x,1,minval)
                    #assert(not np.isnan(A[qi,ki]))
            m[:] = -pre * np.sum((q[:-1]*phi[:-1]*S[:-1]*A[:,:-1] + q[1:]*phi[1:]*S[1:]*A[:,1:])/2 * np.diff(q), axis=-1)
        return ker
