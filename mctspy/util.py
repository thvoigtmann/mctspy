import numpy as np
import scipy

class CorrelatorStack(list):
    def solve_all (self, callback=None, stop_on_zero=False,
                                        stop_condition=None):
        """Solve all correlators in the list.

        This method calls the solver for each correlator in the list,
        iterating through the blocks. For the first block, initialization
        and the solver are called for the first half, and then for each
        second half of a block, the loop is to call the solver, and then
        decimize.

        Parameters
        ----------
        callback : callable, optional
            If set, should be a function taking four arguments: the current
            block number, the first index in the block, the last index,
            and the list of correlators currently being solved.
            This is called after the solution of each half-block, but before
            decimation, so that for example the currently obtained solutions
            can be stored.
        stop_on_zero : bool, default: False
            If set, the solver-loop will stop as soon as all correlators
            are zero in the last half-block that has been solved.
        stop_condition : callable, optional
            If set, this function will be called with the correlator
            object as an argument, and shall return a boolean indicating
            whether to stop or not. This test is performed in addition
            to the stop_on_zero test.

        Notes
        -----
        The iteration over the blocks is the outer loop, so that each
        correlator/model combination can rely on the fact that the base models
        and correlator objects store the solution arrays for the current
        block. They thus also work even if the full solutions are never stored.
        """ 
        if not len(self): return
        blocksize = self[0].blocksize
        halfblocksize = self[0].halfblocksize
        blocks = self[0].blocks
        for _phi_ in self:
            _phi_.solve_first()
        if callback is not None:
            callback (0, 0, halfblocksize, self)
        for d in range(blocks):
            for _phi_ in self:
                _phi_.solve_next (d)
            if callback is not None:
                callback (d, halfblocksize, blocksize, self)
            stop = 0
            for _phi_ in self:
                _phi_.decimize ()
                if stop_on_zero \
                    and np.isclose(_phi_.phi_[halfblocksize],0).all() \
                    and np.isclose(_phi_.phi_[-1],0).all():
                    stop += 1
                elif stop_condition is not None \
                    and stop_condition(_phi_):
                    stop += 1
            if stop == len(self):
                break


def regula_falsi(f,x0,x1,accuracy=1e-8,maxiter=10000,isclose=np.isclose,fargs=()):
    xa, xb = x0, x1
    if x1 > x0:
        xa, xb = x1, x0
    fa, fb = f(xa,*fargs), f(xb,*fargs)
    dx = xb-xa
    iterations = 0
    while (not iterations or (dx > accuracy and iterations <= maxiter)):
        xguess = xa - dx/(fb-fa) * fa
        fguess = f(xguess,*fargs)
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
        if isclose(fguess,0.,rtol=accuracy,atol=accuracy):
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
        Value of the exponent parameter, must be in the range (0.5,1).

    Returns
    -------
    a, b : float, float
        Values of the critical exponents.
    """
    return regula_falsi(lambda x:lambda_func(x,lambda_val),0.2,0.3), \
           -regula_falsi(lambda x:lambda_func(x,lambda_val),-2.0,0.0)

__Bll__ = np.array(
       [0.5 , 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.6 ,
       0.61, 0.62, 0.63, 0.64, 0.65, 0.66, 0.67, 0.68, 0.69, 0.7 , 0.71,
       0.72, 0.73, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79, 0.8 , 0.81, 0.82,
       0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89, 0.9 , 0.91, 0.92, 0.93,
       0.94, 0.95, 0.96, 0.97, 0.98])
__BlB__ = np.array(
       [ -0.22757371,  -0.24058533,  -0.25424881,  -0.26860318,
        -0.28369074,  -0.29955754,  -0.31625371,  -0.33383384,
        -0.35235775,  -0.37189092,  -0.39250509,  -0.41427939,
        -0.43730097,  -0.46166625,  -0.48748195,  -0.51486689,
        -0.54395352,  -0.57488969,  -0.60784165,  -0.64299641,
        -0.68056536,  -0.7207882 ,  -0.76393825,  -0.81032826,
        -0.86031779,  -0.91432265,  -0.97282631,  -1.03639391,
        -1.105691  ,  -1.18150586,  -1.26478111,  -1.35665211,
        -1.45849894,  -1.57201594,  -1.69930869,  -1.84302405,
        -2.00653588,  -2.19421277,  -2.41180743,  -2.66704414,
        -2.97052605,  -3.33715543,  -3.78844805,  -4.35633345,
        -5.0894233 ,  -6.06266095,  -7.38786702,  -9.20098598,
       -11.52443239])

def Blambda(lambda_val):
    r"""Return the coefficient B for the MCT scaling equation.

    Parameters
    ----------
    lambda_val : float
        Value of the exponent parameter.

    Returns
    -------
    B : float
        Value of the coefficient B(lambda) or `np.nan` if out of range.

    Notes
    -----
    The returned values are interpolations of a pre-computed table,
    currently for lambda values [0.5,0.98].
    If `lambda_val` is outside this precalculated range, `np.nan` is returned.
    """
    return np.interp(lambda_val, __Bll__, __BlB__,left=np.nan,right=np.nan)



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
