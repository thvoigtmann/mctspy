.. mctspy documentation master file, created by
   sphinx-quickstart on Mon Sep  9 11:42:31 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

mctspy: Mode-Coupling Theory Solver in Python
=============================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:


This module provides a python implementation of numerical routines for the
mode-coupling theory of the glass transition (MCT). In short, MCT is a
theory that predicts dynamical correlation functions for highly viscous
fluids, as solutions of certain integro-differential equations that are
solved numerically. `mctspy` supports this numerical solution alongside a
number of utility helpers that are often used in conjunction with MCT.

More information on MCT can be found in the standard references.

The core functionality of `mctspy` is the numerical treatment of evolution
equations of the form

.. math::

    \tau_0(q)\partial_t\phi(q,t) + \phi(q,t)
    + \int_0^t m(q,t-t')\partial_{t'}\phi(q,t') = 0

solved as an initial value problem for the unknown :math:`\phi(q,t)` at
a set of given indices :math:`q` (in physical terms: the density correlation
function to a given wave number), where the memory kernel :math:`m(q,t)`
is itself a (usually nonlinear) functional of the sought-after solution,

.. math::

    m(q,t)=\mathcal F[\phi(q,t)]

The nonlinearity of this self-consistent closure causes a bifurcation
scenario, where the long-time limit :math:`f(q)=\lim_{t\to\infty}\phi(q,t)`
can show discontinuities upon a smooth change of the parameters entering
the equations. Close to such a bifurcation, the relaxation times of both
the solution and the memory kernel become arbitrarily large, and hence
adapted numerical integration schemes are needed to treat the MCT equations.

At the moment, `mctspy` is not yet a fully performant solver. It makes
heavy use for the `numba` just-in-time compiler, but this leads to relatively
long warm-up times when running even simple code. This code base is mostly
intended to provide an easy entry point for people wanting to understand,
use, modify and extend the core numerics behind MCT.


.. automodule:: mctspy

Structure Factors
=================

.. autoclass:: mctspy.structurefactors.hssFMT2d
    :members:
    :inherited-members:

Solvers
=======

.. autoclass:: mctspy.non_gaussian_parameter
    :members:
    :inherited-members:

    The MCT equations for the non-Gaussian parameter :math:`\alpha_2(t)` are
    (in 3D, and for Brownian dynamics) written in terms of the function
    :math:`a(t)=(1-\alpha_2(t))(\delta r^2(t))^2` where
    :math:`\delta r^2(t)` is the mean-squared displacement, which needs
    to be calculated first and separately. Then, this class allows to solve

    .. math::

        \begin{multline}
        \partial_ta(t)+D_0^s\int_0^tm^s_0(t-t')\partial_{t'}a(t')\,dt'\\
        =12D_0^s\delta r^2(t)+6D_0^2\int_0^t\tilde m^s_0(t-t')
        \partial_{t'}\delta r^2(t')\,dt'
        \end{multline}

    where the memory kernels are given by the specific model.
    For example, the simple-liquid model in 3d specifies

    .. math::

        \begin{align}
        m^s_0&=\frac1{6\pi^2}\int\rho S_k(c_k^s)^2k^4f_kf_k^s\,dk\\
        \tilde m^s_0&=\frac1{10\pi^2}\int\rho S_k(c_k^s)^2k^4
            (\partial_k^2f^s_k+(2/3k)\partial_kf^s_k)\,dk
        \end{align}

    The memory kernel :math:`m^s_0(t)` is exactly that of the MSD.
    For technical reasons, the implementation expects the model to
    define a two-correlator model, comprised of the two memory kernels
    needed by the NGP equation. The first "correlator" entry in the
    solutions will be the quantity :math:`a(t)` while the second will be
    a copy of the base model's MSD, so that :math:`\alpha_2(t)` can be
    reconstructed for finite times. For very small times, expect this to
    be inaccurate due to cancellation errors; it is best to restrict the
    temporal range where the data is plotted.


Models
======

.. autoclass:: mctspy.tagged_particle_ngp
    :members:
    :inherited-members:

    In essence, we define the following two memory kernels that correspond
    to those needed for the MCT equations for the NGP:

    .. math::

        \begin{align}
        m^s_0&=\frac1{6\pi^2}\int\rho S_k(c_k^s)^2k^4f_kf_k^s\,dk\\
        \tilde m^s_0&=\frac1{10\pi^2}\int\rho S_k(c_k^s)^2k^4
            (\partial_k^2f^s_k+(2/3k)\partial_kf^s_k)\,dk
        \end{align}

    Here, :math:`f^s_k` is the tagged-particle correlator, obtained from
    `self.base` which is set to `model_base.base.base` by the initialization
    method of this class; :math:`f_k` is the collective correlator,
    obtained from `model_base.base.base.base`.

    The first memory kernel is the one that also appears in the MSD,
    however we re-calculate it here. Note however that the MSD is not
    re-calculated (for the sake of modularity, since calculation of the MSD
    is kept to its separate solver).

.. autoclass:: mctspy.simple_liquid_model_2d
    :members:
    :inherited-members:

    The MCT memory kernel implemented here is of the form

    .. math::

        m_q = \rho\int\frac{d^dk}{(4\pi)^d}\frac{S_q}{q^{d+2}}\Omega_{d-1}
              \int_0^\infty dk\,k\Phi_k\int_{|q-k|}^{q+k}dp\,p\Phi_p
              \frac{((q^2+k^2-p^2)c_k + (q^2+p^2-k^2)c_p)^2}
                   {[4q^2k^2-(q^2+k^2-p^2)^2]^{(3-d)/2}}

    specifically for :math:`d=2`. Here, :math:`\Omega_d=2\pi^{d/2}/\Gamma(d/2)`
    is the volume of the unit sphere, and :math:`\Omega_1=2`. Note that
    :math:`\Phi_k=S_k\phi_k(t)` is the non-normalized correlator.

    The code follows the paper by Caraglio et al (2020).
    The kernel method calculates the outer integral following the
    trapezoidal rule,

    .. math::

        q_i^2 m_i = \frac{\rho S_i}{8\pi^2q_i^2}\sum_{j=0}^{M-2}
              \frac{q_j\Phi_jA_{ij}+q_{j+1}\Phi_{j+1}A_{i,j+1}}{2}
              (q_{j+1}-q_j)

    with the inner integral

    .. math::

        \begin{align}
        A_{ij}&=\int_{|q_i-q_j|}^{q_i+q_j}dp\,p\Phi_p
        \frac{[(q_i^2+q_j^2-p^2)c_j+(q_i^2+p^2-q_j^2)c_p]^2}
             {\sqrt{4q_i^2q_j^2-(q_i^2+q_j^2-p^2)^2}} \\
        &= 2q_i^2q_j^2\int_{-1}^1dx\,\Phi_{p(x)}
        \frac{[x(c_j-c_{p(x)})+(q_i/q_j)c_{p(x)}]^2}{\sqrt{1-x^2}}
        \end{align}

    after substitution :math:`x=(q_i^2+q_j^2-p^2)/(2q_iq_j)`.
    The key of the integration method is to perform the inner integral
    in a way that acknowledges the square-root singularity in the
    denominator: the functions of :math:`p(x)` are approximated as
    piece-wise linear given by the grid points, and the resulting integrands
    of the form :math:`x^\alpha/\sqrt{1-x^2}` are calculated analytically.
    Specifically, one needs here

    .. math::

        \begin{align}
        \int dx\,\frac1{\sqrt{1-x^2}} &= \arcsin x\\
        \int dx\,\frac{x}{\sqrt{1-x^2}} &= -\sqrt{1-x^2}\\
        \int dx\,\frac{x^2}{\sqrt{1-x^2}} &= (1/2)\arcsin x-(x/2)\sqrt{1-x^2}\\
        \int dx\,\frac{x^3}{\sqrt{1-x^2}} &= -(1/3)\sqrt{1-x^2}(2+x^2)
        \end{align}

    Care is taken to extend the integrals to the interval :math:`x\in[-1,1]`,
    except where the upper cutoff of the wave-number grid is reached.

    The idea to perform the weighted integrals over piece-wise linear
    functions analytically goes back to Filon and Tuck who used it for the
    Fourier transform.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
