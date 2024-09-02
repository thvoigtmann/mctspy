import numpy as np

from numba import njit

from .__util__ import void

@njit
def _fsolve (f, m, W, phi0, kernel, M, accuracy, maxiter):
    iterations = 0
    converged = False
    newf = phi0
    while (not converged and iterations < maxiter):
        iterations+=1
        f[0] = newf
        kernel (m[0], f[0], 0, 0.)
        newf = m[0] / (W+m[0]) * phi0
        if np.isclose (newf, f[0], rtol=accuracy, atol=0.0).all():
            converged = True
            f[0] = newf
        if not iterations%100: print("iter",iterations)

class nonergodicity_parameter (object):
    def __init__ (self, model, accuracy=1e-12, maxiter=1000000):
        self.model = model
        self.accuracy = accuracy
        self.maxiter = maxiter
        self.f = np.zeros((1,len(model)))
        self.m = np.zeros((1,len(model)))
        self.model.set_base(self.f)
        self.jit_kernel = model.get_kernel(self.m,self.f,0,0.0)

    def solve (self):
        _fsolve(self.f, self.m, self.model.Wq(), self.model.phi0(), self.jit_kernel, len(self.model), self.accuracy, self.maxiter)


@njit
def _esolve(e, dm, f, M, maxiter, accuracy):
    iterations = 0
    converged = False
    newe = np.ones(M)
    while (not converged and iterations < maxiter):
        e[:] = newe
        #newe = (1-f)*model.dm(f,self.e)*(1-f)
        dm(newe,f,e)
        norm = np.sqrt(np.dot(newe,newe))
        if norm>1e-10: newe = newe/norm
        if np.isclose (newe, e, rtol=accuracy, atol=0.0).all():
            converged = True
            e[:] = newe
            eval_ = norm
    return eval_

@njit
def _ehatsolve(ehat, dmhat, f, M, maxiter, accuracy):
    iterations = 0
    converged = False
    newehat = np.ones(M)
    while (not converged and iterations < maxiter):
        ehat[:] = newehat
        dmhat(newehat,f,ehat)
        norm = np.sqrt(np.dot(newehat,newehat))
        if norm>1e-10: newehat = newehat/norm
        if np.isclose (newehat,ehat,rtol=accuracy,atol=0.0).all():
            converged = True
            ehat[:] = newehat
            eval2_ = norm
    return eval2_

class eigenvalue (object):

    def __init__ (self, nep, accuracy=1e-12, maxiter=1000000):
        self.nep = nep
        self.accuracy = accuracy
        self.maxiter = maxiter
        nep.model.set_C (self.nep.f[0])
        f, m = self.nep.f[0], self.nep.m[0]
        self.dm = self.nep.model.get_dm(m,f,f)
        self.dmhat = self.nep.model.get_dmhat(m,f,f)
        self.dm2 = self.nep.model.get_dm2(m,f,f)

    def solve (self):
        f = self.nep.f[0]
        model = self.nep.model
        self.e = np.zeros(len(model))
        self.ehat = np.zeros(len(model))
        self.eval = _esolve(self.e, self.dm, f, len(model), self.maxiter, self.accuracy)
        self.eval2 = _ehatsolve(self.ehat, self.dmhat, f, len(model), self.maxiter, self.accuracy)
        if self.eval > 0:
            dq = model.dq()
            nl = np.dot(dq * self.ehat, self.e)
            nr = np.dot(dq * self.ehat, self.e*self.e * (1-f))
            self.e = self.e * nl/nr
            self.ehat = self.ehat * nr/(nl*nl)
            #self.lam = np.dot(self.ehat, (1-f)*self.dm2(f,self.e)*(1-f))
            C = np.zeros(len(model))
            self.dm2(C,f,self.e)
            self.lam = np.dot(dq * self.ehat, C)
        else:
            self.lam = 0.0


