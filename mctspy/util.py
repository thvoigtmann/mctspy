import numpy as np
import scipy
from .solver import _decimize

class CorrelatorStack(list):
    def solve_all (self, callback=lambda d,i1,i2,corr:None):
        if not len(self): return
        blocksize = self[0].blocksize
        halfblocksize = self[0].halfblocksize
        blocks = self[0].blocks
        for _phi_ in self:
            _phi_.solve_first()
        callback (0, 0, halfblocksize, self)
        for d in range(blocks):
            for _phi_ in self:
                _phi_.solve_next (d)
            callback (d, halfblocksize, blocksize, self)
            for _phi_ in self:
                _phi_.decimize ()


def regula_falsi(f,x0,x1,accuracy=1e-8,maxiter=10000):
    xa, xb, fa, fb = x0, x1, f(x0), f(x1)
    dx = xb-xa
    iterations = 0
    while (not iterations or (dx > accuracy and iterations <= maxiter)):
        xguess = xa - dx/(fb-fa) * fa
        fguess = f(xguess)
        if xguess < xa:
            xb = xa
            xa = xguess
            fb = fa
            fa = fguess
        elif xguess > xb:
            xa = xb
            xb = xguess
            fa = fb
            fb = fguess
        elif ((fguess>0 and fa<0) or (fguess<0 and fa>0)):
            # f(xguess) and f(a) have opposite signs
            # then there must be a root in (a,xguess) if f(x) is continuous
            xb = xguess
            fb = fguess
        else:
            # f(b) and f(xguess) have opposite signs
            # then there must be a root in (xguess,b)
            xa = xguess
            fa = fguess
        if np.isclose(fguess,0.,rtol=accuracy,atol=accuracy):
            break
        dx = xb-xa
        iterations+=1
    return xguess

def lambda_func(x,lambda_val):
    xc = 1 - x
    if xc > 0:
        g = np.exp(scipy.special.loggamma(xc))
    else:
        g = -np.pi/(np.sin(np.pi*xc)*xc)/np.exp(scipy.special.loggamma(-xc))
    return g*(g/np.exp(scipy.special.loggamma(xc-x)))-lambda_val


def exponents(lambda_val):
    r"""Return the exponents given by MCT exponent parameter :math:`\lambda`.

    Parameters
    ----------
    lambda_val : float
        Value of the exponent parameter, must in the range (0.5,1)

    Returns
    -------
    a, b : float, float
        Values of the critical exponents.
    """
    return regula_falsi(lambda x:lambda_func(x,lambda_val),0.2,0.3), \
           -regula_falsi(lambda x:lambda_func(x,lambda_val),-2.0,0.0)


def filon_integrate(f,x,G0,G1):
    r"""Filon-Tuck integration routine for weighted integrals.

    Integrates the function :math:`f(x)` given on a fixed grid of
    points :math:`x` weighted with a function :math:`g(x)` for which the
    integral as well as the integral over :math:`xg(x)` are known
    analytically.

    Parameters
    ----------
    f : array_like
        Function values on the grid `x`
    x : array_like
	Points corresponding to the function values `f`.
    G0 : callable
        Analytic implementation of the integral over the weight function.
    G1 : callable
        Analytic implementation of the integral over x times weight function.

    Notes
    -----
    The integration is performed as

    .. math::

        \int_{x_i}^{x_{i+1}}f(x)g(x)\,dx =
        a \int_{x_i}^{x_{i+1}} g(x)\,dx + b \int_{x_i}^{x_{i+1}} xg(x)\,dx

    and it is assumed that the integrals over :math:`g(x)` and
    :math:`xg(x)` are known analytically and provided as arguments to
    this function.

    Setting G0=lambda x:x and G1=lambda x:x*x/2 evaluates the integral
    over f following the ordinary trapezoidal rule.

    Setting G0=lambda x:np.sin(x) and G1=lambda x:x*np.sin(x)+np.cos(x)
    evaluates the Fourier-cosine integral over f following the method
    suggested by Tuck.
    """
    b = np.diff(f,axis=-1)/np.diff(x,axis=-1)
    a = f[...,:-1] - x[...,:-1]*b
    return np.sum(a*np.diff(G0(x)) + b*np.diff(G1(x)),axis=-1)


def filon_cos_transform(f,x,w):
    """Filon cosine transform.

    Parameters
    ----------

    f : array_like, shape (M,N)
        Function values on the grid `x`.
    x : array_like, shape (M,N)
    w : array_like, shape (M,N)
    """
    return filon_integrate(f,x,lambda x:np.sin(w*x)/w,
                               lambda x:x*np.sin(w*x)/w+np.cos(w*x)/w**2)
def filon_sin_transform(f,x,w):
    """Filon sine transform.
    """
    return filon_integrate(f,x,lambda x:-np.cos(w*x)/w,
                               lambda x:-x*np.cos(w*x)/w+np.sin(w*x)/w**2)
