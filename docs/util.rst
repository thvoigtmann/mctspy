Utilities
=========

MCT exponents
-------------

The asymptotic analysis of MCT close to regular glass-transition points
(the :math:`\mathcal A_2` bifurcation scenario) yields power laws of the
form :math:`\phi(q,t)\simeq f^c(q)+h(q)(t/t_0)^{-a}` for the decay towards
the nonergodic plateau, and :math:`\phi(q,t)\simeq f^c(q)-Bh(q)(t/t_\sigma)^b`
for the decay from the plateau. The exponents are fixed by a single parameter
:math:`\lambda` that can be calculated from the memory kernel at the
critical point knowing the critical nonergodicity parameters; see
:py:func:`mctspy.eigenvalue.solve()`.

The exponent fixes :math:`a>0` and :math:`b>0` by the relation

.. math::

    \frac{\Gamma(1-a)^2}{\Gamma(1-2a)} = \lambda
    = \frac{\Gamma(1+b)^2}{\Gamma(1+2b)|}

which results in :math:`0<a\lesssim0.395` and :math:`0<b<1` since
:math:`\lambda\in[1/2,1[`.

.. autofunction:: mctspy.exponents


Filon Fourier-Laplace Transforms
--------------------------------

A common post-processing of correlation functions obtained by MCT is
to calculate their spectra, by the one-sided Fourier-Laplace transform.
Since the correlator is stored on a non-regular grid spanning many decades
in time, this requires some thought. We provide routines to calculate
these integrals following a simple but powerful integration method commonly
referred to as Filon integration, but actually in the simplified form given
by [Tuck1967]_

.. [Tuck1967] E. O. Tuck, Math. Comp. 21, 239 (1967),
   `DOI:10.1090/s0025-5718-67-99892-4 <https://doi.org/10.1090/s0025-5718-67-99892-4>`_

.. autofunction:: mctspy.util.filon_integrate

.. autofunction:: mctspy.util.filon_cos_transform
.. autofunction:: mctspy.util.filon_sin_transform

These are special functions using :py:func:`mctspy.util.filon_integrate`
for calculating the one-sided cosine- and sine-transforms
(Fourier-Laplace transforms) of temporal data.

The functions vectorize, so that you can calculate the entire spectrum
of a function for a set of frequences like this (assuming that `t` and `f`
are the time and function values on a matching grid, and `w` is an array
for frequencies):

.. code-block:: python

    t_,w_ = np.meshgrid(t,w)
    f_,_ = np.meshgrid(f,w)
    fw = mct.util.filon_cos_transform(f_,t_,w_)
