import numpy as np
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


# for these, f,x,w need to have shape (M,N) where N is the number of
# time-domain data points, and M the number of frequency data points
# ie use x_,w_ = np.meshgrid(x,w)
# f_,_ = np.meshgrid(f,w)
def filon_cos_transform(f,x,w):
    return filon_integrate(f,x,lambda x:np.sin(w*x)/w,
                               lambda x:x*np.sin(w*x)/w+np.cos(w*x)/w**2)
def filon_sin_transform(f,x,w):
    return filon_integrate(f,x,lambda x:-np.cos(w*x)/w,
                               lambda x:-x*np.cos(w*x)/w+np.sin(w*x)/w**2)
