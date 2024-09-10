Correlators and Solvers
=======================

Simple Liquid-Like
------------------

The "simple liquid-like" MCT equations are defined by

.. math::

    \tilde\tau_0(q)\partial_t\phi(q,t) + q^2\phi(q,t)
    + \int_0^t q^2m(q,t-t')\partial_{t'}(q,t')\,dt' = 0

with a scalar correlation function :math:`\phi(q,t)` that depends on
the magnitude of the wave vector, :math:`q=|\vec q|`.

.. autoclass:: mctspy.correlator
    :members:
    :inherited-members:

.. autoclass:: mctspy.mean_squared_displacement
    :members:
    :inherited-members:

    The MCT equation for the mean-squared displacement :math:`\delta r^2(t)`
    are (in 3D, and for Brownian dynamics) is

    .. math::

        \partial_t\delta r^2(t)
        + D_0^s\int_0^tm^s_0(t-t')\partial_{t'}\delta r^2(t')\,dt'=6D_0^s

    where the memory kernel is given by the :math:`q\to0` limit of a
    specific model, for example :py:class:`mctspy.tagged_particle_q0`.

.. autoclass:: mctspy.non_gaussian_parameter
    :members:
    :inherited-members:

    The MCT equation for the non-Gaussian parameter :math:`\alpha_2(t)` is
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
    For example, the simple-liquid model in 3d specifies, via
    :py:class:`mctspy.tagged_particle_ngp`,

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




