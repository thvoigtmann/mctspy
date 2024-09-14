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


Mixtures
--------

.. autoclass:: mctspy.mixture_model
    :members:
    :inherited-members:

    The MCT memory-kernel expression for mixtures is given by

    .. math::

        M_{\alpha\beta}(q)=\frac{\varrho}{2q^4}\int\frac{d^3k}{(2\pi)^3}
        \sum_{\alpha'\alpha''\beta'\beta''}
        V_{\alpha\alpha'\alpha''}(\vec q,\vec k\vec p)
        V_{\beta\beta'\beta''}(\vec q,\vec k\vec p)
        F_{\alpha'\beta'}(k)F_{\alpha''\beta''}(p)

    with :math:`\vec q=\vec k+\vec p` and vertices

    .. math::

        V_{\alpha\alpha'\alpha''}(\vec q,\vec k\vec p)
        =(\vec q\vec k)\tilde c_{\alpha\alpha'}(k)\delta_{\alpha\alpha''}
        +(\vec q\vec p)\tilde c_{\alpha\alpha''}(p)\delta_{\alpha\alpha'}
        +q\sqrt{x_\alpha}\tilde c^{(3)}_{\alpha\alpha'\alpha''}(\vec q,\vec k)

    where we have introduced the modified DCF
    :math:`\tilde c_{\alpha\beta}(k)=\sqrt{x_\beta}c_{\alpha\beta}(k)`
    and :math:`\tilde c^{(3)}_{\alpha\beta\gamma}(\vec q,\vec k)
    =\sqrt{x_\beta x_\gamma}c^{(3)}_{\alpha\beta\gamma}(\vec q,\vec k)`.
    The quantities :math:`\boldsymbol F` appearing in the expression are
    assumed to be "normalized" to correspond to :math:`\boldsymbol S`, the
    static structure factor, if taken as equal-time averages.

    The three-particle static correlation function is currently not
    implemented.

    We restrict wave vectors to a finite grid, :math:`q_i = (\Delta q)(i+1/2)`
    and write the discretized integral with
    :math:`M_{\alpha\beta}(q)=(1/q^2)\hat M_{\alpha\beta}(q)`

    .. math::
        \hat M_{\alpha\beta}(q) = \frac{(\Delta q)^2}{32\pi^2q^3}\sum_{kp}
        \sum_{\alpha'\alpha''\beta'\beta''} kp
        \hat V_{\alpha\alpha'\alpha''}(q,k,p)\hat V_{\beta\beta'\beta''}(q,k,p)
        F_{\alpha'\beta'}(k)F_{\alpha''\beta''}(p)

    with

    .. math::

        \hat V_{\alpha\alpha'\alpha''}(q,k,p) = \left[
        \delta_{\alpha\alpha''}(q^2+k^2-p^2)\hat c_{\alpha\alpha'}(k)
        +\delta_{\alpha\alpha'}(q^2-k^2+p^2)\hat c_{\alpha\alpha''}(p)\right]

    and :math:`\hat c_{\alpha\beta}(k)=\sqrt{\varrho_\beta}c_{\alpha\beta}(k)`.

    In the above expression and throughout our mixture-model
    implementation, we normalize the correlation functions to have as their
    initial value the non-singular part of the structure factor, i.e. the
    one whose diagonal matrix elements approach unity for large :math:`q`.
    This differs from the common definition of a partial dynamic density
    correlation function: one has to multiply the :math:`\alpha\beta`
    matrix element of our correlation functions by
    :math:`\sqrt{x_\alpha x_\beta}`, and the memory function has to be divided
    by that expression when transforming to the common notation.

    The model also stores :math:`\hat M` instead of :math:`M`.

    These conventions are consistent throughout the rest of the code.

    The memory-kernel integration here uses the Bengtzelius trick. We have

    .. math::

        \begin{align}
        \hat M_{q,\alpha\beta}[f,g]&=\nu
        \sum_{n=1}^3\Big[\phantom+\,
        \sum_{\alpha'\beta'}
        \sum_k a_{\alpha\alpha'\beta\beta'}^{(n)}(k)f_{k,\alpha'\beta'}
        \sum_{p(q,k)} b^{(n)}(q,p)g_{p,\alpha\beta}\\ 
        &\phantom{=p\sum_{n=1}^3\Big[}+
        \sum_{\alpha''\beta''}
        \sum_k a^{(n)}(k)f_{k,\alpha\beta}
        \sum_{p(q,k)}
          b_{\alpha\alpha''\beta\beta''}^{(n)}(q,p)g_{p,\alpha''\beta''}
        \\
        &\phantom{=\nu\sum_{n=1}^3\Big[}+
        \sum_{\alpha'\beta''}
        \sum_k a_{\alpha\alpha'}^{(n)}(k)f_{k,\alpha'\beta}
        \sum_{p(q,k)} b_{\beta\beta''}^{(n)}(q,p)g_{p,\alpha\beta''}\\
        &\phantom{=\nu\sum_{n=1}^3\Big[}+
        \sum_{\alpha''\beta'}
        \sum_k a_{\beta\beta'}^{(n)}(k)f_{k,\alpha\beta'}
        \sum_{p(q,k)} b_{\alpha\alpha''}^{(n)}(q,p)g_{p,\alpha''\beta}
        \;\phantom+\Big]
        \end{align}

    with :math:`\nu = (\Delta q)^2/(32\pi^2q^3)` as a prefactor. This
    expression does not assume symmetry under swapping :math:`f` with
    :math:`g` and is used for the calculation of the stability matrix.
    For the case of the memory kernel itself, the symmetry allows to further
    reduce to

    .. math::

        \begin{align}
        \hat M_{q,\alpha\beta}&=2\nu\sum_{n=1}^3
        \sum_{\gamma\delta}\Big[\phantom+\,
        \sum_k a_{\alpha\gamma\beta\delta}^{(n)}(k)\Phi_{k,\gamma\delta}
        \sum_{p(q,k)} b^{(n)}(q,p)\Phi_{p,\alpha\beta}\\ \nonumber
        &\phantom{=2\nu\sum_{n=1}^3\sum_{\gamma\delta}\Big[}+
        \sum_k a_{\alpha\gamma}^{(n)}(k)\Phi_{k,\gamma\beta}
        \sum_{p(q,k)} b_{\beta\delta}^{(n)}(q,p)\Phi_{p,\alpha\delta}
        \;\phantom+\Big]
        \end{align}

    In the above expressions, the Bengtzelius coefficients are

    .. math::

        \begin{align}
        a^{(1)}(k)&=k^5
        & a_{\alpha\beta}^{(1)}(k)&=k\tilde c_{\alpha\beta}(k)
        & a_{\alpha\beta\gamma\delta}^{(1)}(k)&=k^5\tilde c_{\alpha\beta}(k)
                                               \tilde c_{\gamma\delta}(k)\\
        a^{(2)}(k)&=-2k^3
        & a_{\alpha\beta}^{(2)}(k)&=-k^5\tilde c_{\alpha\beta}(k)
        & a_{\alpha\beta\gamma\delta}^{(2)}(k)&=2k^3\tilde c_{\alpha\beta}(k)
                                                \tilde c_{\gamma\delta}(k)\\
        a^{(3)}(k)&=k
        & a_{\alpha\beta}^{(3)}(k)&=2k^3\tilde c_{\alpha\beta}(k)
        & a_{\alpha\beta\gamma\delta}^{(3)}(k)&=k\tilde c_{\alpha\beta}(k)
                                             \tilde c_{\gamma\delta}(k)
        \\
        b^{(1)}(q,p)&=p
        & b_{\alpha\beta}^{(1)}(q,p)&=p(q^4\!-\!p^4)\tilde c_{\alpha\beta}(p)
        & b_{\alpha\beta\gamma\delta}^{(1)}(q,p)&=p\tilde c_{\alpha\beta}(p)
                                               \tilde c_{\gamma\delta}(p)\\
        b^{(2)}(q,p)&=p(q^2\!-\!p^2)
        & b_{\alpha\beta}^{(2)}(q,p)&=p\tilde c_{\alpha\beta}(p)
        & b_{\alpha\beta\gamma\delta}^{(2)}(q,p)&=p(q^2\!+\!p^2)
                      \tilde c_{\alpha\beta}(p)\tilde c_{\gamma\delta}(p)\\
        b^{(3)}(q,p)&=p(q^2\!-\!p^2)^2
        & b_{\alpha\beta}^{(3)}(q,p)&=p^3\tilde c_{\alpha\beta}(p)
        & b_{\alpha\beta\gamma\delta}^{(3)}(q,p)&=p(q^2\!+\!p^2)^2
                      \tilde c_{\alpha\beta}(p)\tilde c_{\gamma\delta}(p)
        \end{align}

    As in the scalar case, the performance gain known as Bengtzelius' trick
    comes from realizing that the inner sum can be reduced to a
    :math:`\mathcal O(1)` operation when its value for the previous
    outer wave number is known.




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
