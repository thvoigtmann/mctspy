MCT Models
==========

Simple Liquids
--------------

"Simple liquid" models are the "standard models" of MCT.

.. autoclass:: mctspy.simple_liquid_model
    :members:
    :inherited-members:


Common Additions to Simple-Liquid Models
----------------------------------------

.. autoclass:: mctspy.tagged_particle_model

.. autoclass:: mctspy.tagged_particle_q0

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


Two-dimensional MCT
-------------------

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



Schematic Models
----------------
.. include:: <isolat1.txt>

.. autoclass:: mctspy.f12model
    :members:
    :inherited-members:

.. autoclass:: mctspy.sjoegren_model
    :members:
    :inherited-members:

.. autoclass:: mctspy.bosse_krieger_model
    :members:
    :inherited-members:

.. autoclass:: mctspy.f12gammadot_model
    :members:
    :inherited-members:

.. autoclass:: mctspy.f12gammadot_tensorial_model
    :members:
    :inherited-members:
