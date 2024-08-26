from numba import njit, carray
from .__util__ import nparray, model_base

class schematic (model_base):
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

class f12model (model_base):
    def __init__ (self, v1, v2):
        self.v1 = v1
        self.v2 = v2
    def __len__ (self):
        return 1
    def make_kernel (self, phi, i=0, t=0.0):
        v1 = self.v1
        v2 = self.v2
        @njit
        def m(phi, i, t):
            return  v1*phi + v2*phi*phi
        return m
    def make_dm (self, phi, dphi):
        v1 = self.v1
        v2 = self.v2
        @njit
        def dm(phi, dphi):
            return v1 * dphi + 2*v2 * phi*dphi
        return dm
    def make_dmhat (self, f, ehat):
        v1 = self.v1
        v2 = self.v2
        @njit
        def dmhat(f, ehat):
            return (1-f)*(v1*ehat + 2*v2 * f*ehat)*(1-f)
        return dmhat
    def make_dm2 (self, phi, dphi):
        v2 = self.v2
        @njit
        def dm2(phi, dphi):
            return v2 * dphi*dphi
        return dm2

class f12gammadot_model (f12model):
    def __init__ (self, v1, v2, gammadot=0.0, gammac=0.1):
        f12model.__init__(self, v1, v2)
        self.gammadot = gammadot
        self.gammac = gammac
    def make_kernel (self, phi, i=0, t=0.0):
        v1 = self.v1
        v2 = self.v2
        gammadot = self.gammadot
        gammac = self.gammac
        @njit
        def ker(phi, i, t):
            gt = gammadot*t / gammac
            return (v1*phi + v2*phi*phi) * 1.0/(1.0 + gt*gt)
        return ker

class sjoegren_model (model_base):
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



class msd_model (model_base):
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
