import numpy as np

from numba import njit

@njit
def _fsolve (f, m, kernel, M, accuracy, maxiter):
    iterations = 0
    converged = False
    newf = np.ones(M)
    while (not converged and iterations < maxiter):
        iterations+=1
        f = newf
        m = kernel (f,0,0.)
        newf = m / (1.0+m)
        if np.isclose (newf, f, rtol=accuracy, atol=0.0).all():
            converged = True
            f = newf
        if not iterations%100: print("iter",iterations,f[0])
    return f,m

class nonergodicity_parameter (object):
    def __init__ (self, model, accuracy=1e-12, maxiter=1000000):
        self.model = model
        self.accuracy = accuracy
        self.maxiter = maxiter
        self.f = np.zeros(len(model))
        self.m = np.zeros(len(model))
        self.jit_kernel = model.get_kernel(self.f)

    def solve (self):
        self.f,self.m = _fsolve(self.f, self.m, self.jit_kernel, len(self.model), self.accuracy, self.maxiter)


@njit
def _esolve(e, dm, f, M, maxiter, accuracy):
    iterations = 0
    converged = False
    newe = np.ones(M)
    while (not converged and iterations < maxiter):
        e[:] = newe
        #newe = (1-f)*model.dm(f,self.e)*(1-f)
        newe = (1-f)*dm(f,e)*(1-f)
        norm = np.sqrt(np.dot(newe,newe))
        if norm>1e-10: newe = newe/norm
        if np.isclose (newe, e, rtol=accuracy, atol=0.0).all():
            converged = True
            e[:] = newe
            eval_ = norm
    return eval_,e

@njit
def _ehatsolve(ehat, dmhat, f, M, maxiter, accuracy):
    iterations = 0
    converged = False
    newehat = np.ones(M)
    while (not converged and iterations < maxiter):
        ehat[:] = newehat
        newehat = dmhat(f,ehat)
        norm = np.sqrt(np.dot(newehat,newehat))
        if norm>1e-10: newehat = newehat/norm
        if np.isclose (newehat,ehat,rtol=accuracy,atol=0.0).all():
            converged = True
            ehat[:] = newehat
            eval2_ = norm
    return eval2_,ehat

class eigenvalue (object):

    def __init__ (self, nep, accuracy=1e-12, maxiter=1000000):
        self.nep = nep
        self.accuracy = accuracy
        self.maxiter = maxiter
        self.dm = self.nep.model.get_dm(self.nep.f,self.nep.f)
        self.dmhat = self.nep.model.get_dmhat(self.nep.f,self.nep.f)
        self.dm2 = self.nep.model.get_dm2(self.nep.f,self.nep.f)

    def solve (self):
        f = self.nep.f
        model = self.nep.model
        self.e = np.zeros(len(model))
        self.ehat = np.zeros(len(model))
        self.eval, self.e = _esolve(self.e, self.dm, f, len(model), self.maxiter, self.accuracy)
        self.eval2, self.ehat = _ehatsolve(self.ehat, self.dm, f, len(model), self.maxiter, self.accuracy)
        if self.eval > 0:
            nl = np.dot(self.ehat, self.e)
            nr = np.dot(self.ehat, self.e*self.e / (1-f))
            self.e = self.e * nl/nr
            self.ehat = self.ehat * nr/(nl*nl)
            self.lam = np.dot(self.ehat, (1-f)*self.dm2(f,self.e)*(1-f))
            #nr = self.ehat * self.e*self.e / (1-f)
            #print ("nr {}".format(nr))
            #self.lam = self.lam / nr
        else:
            self.lam = 0.0


