from numba import njit, carray
from .__util__ import nparray

class schematic (object):
    def __init__ (self, func, M=1):
        self.func = func
        self.M = M
    def __len__ (self):
        return self.M
    def make_kernel (self, phi, i=None, t=0.0):
        F = self.func
        if i is None:
            @njit
            def ker(phi):
                return F(phi)
        else:
            @njit
            def ker(phi, i, t):
                return F(phi)
        return ker
    def m (self, phi=0, i=0, t=0):
        return self.func(phi)

class f12model (object):
    def __init__ (self, v1, v2):
        self.v1 = v1
        self.v2 = v2
    def __len__ (self):
        return 1
    def make_kernel (self, phi, i=None, t=0.0):
        v1 = self.v1
        v2 = self.v2
        if i is None:
            @njit
            def ker(phi):
                return  v1*phi + v2*phi*phi
        else:
            @njit
            def ker(phi, i, t):
                return  v1*phi + v2*phi*phi
        return ker
    def m (self, phi=0, i=0, t=0):
        return self.v1 * phi + self.v2 * phi*phi
    def dm (self, phi, dphi):
        return self.v1 * dphi + 2 * self.v2 * phi*dphi
    def dmhat (self, f, ehat):
        return (1-f)*self.dm(f,ehat)*(1-f)
    def dm2 (self, phi, dphi):
        return self.v2 * dphi*dphi

class f12gammadot_model (object):
    def __init__ (self, v1, v2, gammadot=0.0, gammac=0.1):
        self.v1 = v1
        self.v2 = v2
        self.gammadot = gammadot
        self.gammac = gammac
    def __len__ (self):
        return 1
    def make_kernel (self, phi, i=None, t=0.0):
        v1 = self.v1
        v2 = self.v2
        gammadot = self.gammadot
        gammac = self.gammac
        if i is None:
            @njit
            def ker(phi):
                return (v1*phi + v2*phi*phi)
        else:
            @njit
            def ker(phi, i, t):
                gt = gammadot*t / gammac
                return (v1*phi + v2*phi*phi) * 1.0/(1.0 + gt*gt)
        return ker
    def m (self, phi, i, t):
        gt = self.gammadot * t / self.gammac
        return (self.v1 * phi + self.v2 * phi*phi) * 1.0/(1.0 + gt*gt)

class sjoegren_model (object):
    def __init__ (self, vs, base_correlator):
        self.vs = vs
        #self.base = base_correlator
        self.base_phi = base_correlator.phi_addr()
    def __len__ (self):
        return 1
    def make_kernel (self, phis, i=None, t=0.0):
        vs = self.vs
        base_phi = self.base_phi
        if i is None:
            @njit
            def ker(phis):
                phi = nparray(base_phi)
                return vs*phi*phis
        else:
            @njit
            def ker(phis, i, t):
                phi = nparray(base_phi)
                return vs * phi[i] * phis
        return ker
    def m (self, phi=0, i=0, t=0):
        return self.vs * phi * self.base.phi[i]



class msd_model (object):
    def __init__ (self, vs, base_correlator):
        self.vs = vs
        #self.base = base_correlator
        self.base_phi = base_correlator.phi_addr()
    def __len__ (self):
        return 1
    def make_kernel (self, phis, i=None, t=0.0):
        vs = self.vs
        base_phi = self.base_phi
        if i is None:
            @njit
            def ker(phis):
                phi = nparray(base_phi)
                return vs*phi*phi
        else:
            @njit
            def ker(phis, i, t):
                phi = nparray(base_phi)
                return vs * phi[i] * phi[i]
        return ker
    def m (self, phi=0, i=0, t=0):
        return self.vs * self.base.phi[i] * self.base.phi[i]
