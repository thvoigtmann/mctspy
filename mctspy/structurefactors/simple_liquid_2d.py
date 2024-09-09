import numpy as np
import scipy

class hssFMT2d (object):
    """Structure factor for 2d hard disks, fundamental measure theory (FMT).

    This implements the expression derived in Thorneywork.2018.
    """
    def __init__ (self, eta):
        self.eta = eta
    def density (self):
        return self.eta*4/np.pi
    def cq (self, q):
        etacmp = (1-self.eta)**2
        j0 = scipy.special.j0(q/2)
        j1 = scipy.special.j1(q/2)
        return (-(5./4)*etacmp*(q*j0)**2 \
            + (4*((self.eta-20)*self.eta+7)+(5./4)*etacmp*q**2)*j1**2 \
            + 2*(self.eta-13)*(1-self.eta)*q*j1*j0) * np.pi/(6*q*q*(1-self.eta)**3)
    def Sq (self, q):
        cq_ = self.cq(q)
        return 1.0 / (1.0 - self.density() * cq_), cq_
