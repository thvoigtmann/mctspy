Static Structure Factors
========================

Simple Liquids (3D)
-------------------

.. autoclass:: mctspy.structurefactors.hssPY
    :members:
    :inherited-members:

    The analytical expression for the direct correlation function used
    here was derived by Wertheim [1]_:

    .. math::

        \begin{align}
        c(q) &= 4\pi\alpha\left(\frac{\cos q}{q^2}-\frac{\sin q}{q^3}\right)
        + 2\pi\alpha\varphi\left(\frac{\cos q}{q^2}-4\frac{\sin q}{q^3}
        - 12\frac{\cos q}{q^4}+24\frac{\sin q}{q^5}-24\frac{1-\cos q}{q^6}
        \right)\\
        &+ 8\pi\beta\left(\frac12\frac{\cos q}{q^2}-\frac{\sin q}{q^3}
        + \frac{1-\cos q}{q^4}\right) \end{align}

    where :math:`\alpha=(1+2\varphi)^2/(1-\varphi)^4` and
    :math:`\beta=-6\varphi(1+\varphi/2)^2/(1-\varphi)^4`.

    For low :math:`q`, an expansion up to :math:`\mathcal O(q^4)` is used:

    .. math::

        c(q)=-\left(\pi\alpha(4+\varphi)/3+\pi\beta\right)
        + q^2\left(\pi\alpha(2/15+\varphi/24)+\pi\beta/9\right)
        - q^4\left(\pi\alpha(1/210+\varphi/600)+\pi\beta/240\right)
        + {\mathcal O}(q^6)


    .. [1] M. S. Wertheim, Phys. Rev. Lett. 10, 321 (1963), `DOI:10.1103/PhysRevLett.10.321 <https://doi.org/10.1103/PhysRevLett.10.321>`_


.. autoclass:: mctspy.structurefactors.hssPYtagged
    :members:
    :inherited-members:

    The expression implemented here is derived from the analytical result
    obtained by Baxter for the hard-sphere binary mixture, taking the limit
    of one species density to zero. In particular,

    .. math::

        \begin{align}
        c^s(q) &= 4\pi\left(A\frac{\cos(q/2)\cos(q\delta/2)-\sin(q/2)
        \sin(q\delta/2)}{q^2}-B\frac{\cos(q\delta/2)\sin(q/2)+\cos(q/2)
        \sin(q\delta/2)}{q^3}-D\frac{\sin(q/2)\sin(q\delta/2)}{q^4}\right)\\
        &-\frac{4\pi a_2}{q^6}\left(q\cos(q/2)-2\sin(q/2)\right)
        \left(q\delta\cos(q\delta/2)-2\sin(q\delta/2)\right)
        \end{align}

    with :math:`A=(1/2)(1-\varphi+\delta(1+2\varphi))/(1-\varphi)^2`,
    :math:`B=((1-\varphi)^2-3\delta\varphi(1+2\varphi))/(1-\varphi)^3`,
    :math:`D=6\varphi(2+\varphi+\delta(1+2\varphi))/(1-\varphi)^3`,
    :math:`a_2=6\varphi/\pi/(1-\varphi)^2(1+6\varphi/(1-\varphi)+9(\varphi/(1-\varphi))^2)`.

    For low :math:`q`, an expansion is used,

    .. math::

        \begin{align}
        c^s(q) &= -\frac{\pi}6\left(3A(1+\delta)^2-\frac{B}{2}(1+\delta)^3
        -\frac{D}{4}\delta(1+\delta^2)+\frac{\delta^3\varphi(1+2\varphi)}
        {(1-\varphi)^4}\right)\\
        &+\frac{\pi}{240}\left(\frac52A(1+\delta)^4-\frac{B}{4}(1+\delta)^5
        -\frac{D}{24}\delta(3+10\delta^2+3\delta^4)
        +\frac{(\delta^3+\delta^5)\varphi(1+2\varphi)^2}{(1-\varphi)^4}
        \right)q^2\\
        &-\frac{\pi}{26880}\left(\frac73A(1+\delta)^6-\frac{B}{6}(1+\delta)^7
        -\frac{D}{12}\delta(1+7\delta^2+7\delta^4+\delta^6)
        +\frac{\delta^3\varphi(5+14\delta^2+5\delta^4)(1+2\varphi)^2}
        {5(1-\varphi)^4}\right)q^4
        \end{align}


Simple Liquids (2D)
-------------------

.. autoclass:: mctspy.structurefactors.hssFMT2d
    :members:
    :inherited-members:
 

Mixtures (3D)
-------------

.. autoclass:: mctspy.structurefactors.hsmPY
    :members:
    :inherited-members:

    This implements the Percus-Yevick (PY) approximation for the
    structure factors of hard-sphere mixtures with a user-determined
    number of species, specified through their number densities
    :math:`\rho_\alpha` and their diameters :math:`d_\alpha`
    (where Greek indices label the species).

    Note that in contrast to the one-component HSS-PY structure factor,
    :py:class:`mctspy.structurefactors.hssPY`, we do not work with
    packing fractions here, but with the number densities.

    The implementation is based on analytic expressions that follow from
    Fourier transforming the real-space expressions derived by Baxter
    [2]_ using a Wiener-Hopf factorization method.
    This gives the DCF :math:`c_{\alpha\beta}(q)` directly, so that the
    structure factor matrix follows from the Ornstein-Zernike equation.
    For small wave numbers, these analytic expressions are expanded to
    yield numerically more stable results

    In detail, the PY approximation for the hard-sphere mixture is thus
    given by

    .. math::

        \begin{align} c_{\alpha\beta}(q) &= -4\pi\left(
        A_{\alpha\beta}\frac{\sin(qd_\alpha/2)\sin(qd_\beta/2)-
        \cos(qd_\alpha/2)\cos(qd_\beta/2)}{q^2}
        \right.\\ &\left.
        +B_{\alpha\beta}\frac{\cos(qd_\alpha/2)\sin(qd_\beta/2)+
        \cos(qd_\beta/2)\sin(qd_\alpha/2)}{q^3}
        \right.\\ &\left.
        +D_{\alpha\beta}\frac{\sin(qd_\alpha/2)\sin(qd_\beta/2)}{q^4}
        \right.\\ &\left.
        +\frac{a_2\pi}{q^4}\left(
        \cos(qd_\alpha/2)\cos(qd_\beta/2)d_\alpha d_\beta
        +\frac{\sin(qd_\alpha/2)\sin(qd_\beta/2)}{q^2}
        -\frac{2(\cos(qd_\alpha/2)\sin(qd_\beta/2)d_\alpha
        +\cos(qd_\beta/2)\sin(qd_\alpha/2)d_\beta)}{q}\right)\right)
        \end{align}

    with parameters

    .. math::

        \begin{align} A_{\alpha\beta}&=\frac{d_{\alpha\beta}(1-\xi_3)
        +(3/2)d_\alpha d_\beta\xi_2}{(1-\xi_3)^2} \\
        B_{\alpha\beta}&=\frac1{1-\xi_3}-\beta_0 d_\alpha d_\beta \\
        D_{\alpha\beta}&=\frac{6\xi_2}{(1-\xi_3)^2}+4d_{\alpha\beta}\beta_0 \\
        a_2&=\frac6\pi\frac{\xi_0}{(1-\xi_3)^2}+\frac{36}\pi\frac{\xi_1\xi_2}
        {(1-\xi_3)^3}+\frac{54}\pi\frac{\xi_2^3}{(1-\xi_3)^4} \\
        \beta_0&=(9\xi_2^2+3\xi_1(1-\xi_3))/(1-\xi_3)^3\end{align}

    and :math:`d_{\alpha\beta}=(d_\alpha+d_\beta)/2`,
    :math:`\xi_n=(\pi/6)\sum_\gamma\varrho_\gamma d_\gamma^n`.

    For low :math:`q`, the implementation here uses an analytical expansion
    to avoid cancellation errors. Specifically,

    .. math::

        \begin{align} c_{\alpha\beta}(q)&=c_{\alpha\beta}^{(0)}
        +\frac12c_{\alpha\beta}^{(2)}q^2+\frac1{4!}c_{\alpha\beta}^{(4)}q^4
        +{\mathcal O}(q^6) \\
        c_{\alpha\beta}^{(0)}&=-\frac\pi6\left(
        \frac\pi6a_2(d_\alpha d_\beta)^3+2(d_\alpha+d_\beta)^2A_{\alpha\beta}
        +\frac12(d_\alpha d_\beta)^2D_{\alpha\beta}\right)\\
        c_{\alpha\beta}^{(2)}&=\frac\pi{60}\left(
        A_{\alpha\beta}(d_\alpha+d_\beta)^4+\frac16D_{\alpha\beta}
        (d_\alpha d_\beta)^3+\left(\frac14D_{\alpha\beta}+\frac\pi{12}a_2
        d_\alpha d_\beta\right)(d_\alpha d_\beta)^2(d_\alpha^2+d_\beta^2)
        \right)\\
        c_{\alpha\beta}^{(4)}&=-\frac\pi{560}\left(
        A_{\alpha\beta}(d_\alpha+d_\beta)^6+\left(\frac14D_{\alpha\beta}
        +\frac\pi{12}a_2d_\alpha d_\beta\right)(d_\alpha d_\beta)^2
        (d_\alpha^4+d_\beta^4)+\frac13\left(D_{\alpha\beta}(d_\alpha
        +d_\beta)^2+\frac{7\pi}{10}a_2(d_\alpha d_\beta)^2\right)
        (d_\alpha d_\beta)^3+\frac16D_{\alpha\beta}(d_\alpha d_\beta)^4
        \right)\end{align}

    Note that the convention we use is

    .. math::

        \boldsymbol S(q) = \left[1 - \boldsymbol\rho^{1/2}\cdot
            \boldsymbol c(q)\cdot\boldsymbol\rho^{1/2}\right]^{-1}

    where :math:`\boldsymbol\rho` is the diagonal matrix with the partial
    number densities as its entries. This implies that the structure-factor
    matrix is normalized such that for large wave number it approaches the
    unit matrix. The conventional static structure factor is obtained
    by multiplying with :math:`\boldsymbol\rho^{1/2}` from both sides again.

    .. [2] R. J. Baxter, J. Chem. Phys. 52, 4559 (1970), `DOI:10.1063/1.1673684 <https://doi.org/10.1063/1.1673684>`_

