Implementation Details
======================


beta-Scaling Solvers
--------------------

The :py:class:`mctspy.beta_scaling_function` solver uses the same
idea of algorithm as used for the correlator, with a few modifications.
First, there is no separate memory kernel; second, the discretized
equation leads to a quadratic equation to be solved at each time step:

.. math::

    -\lambda g_i^2 + 2\,dG_1\,g_i + C = 0

The standard iteration scheme does not work well for this equation,
especially when :math:`g_i` becomes negative. Older versions of the
MCT solver code used a regula falsi scheme (this is still implemented
in the code, but not currently used). It turns out, that a direct
solution of the quadratic equation seems to work well.

In the case of SBR, the equation is modified to

.. math::
    -\lambda g_{x,i}^2 + 2(dG_{x,1} + d\alpha/dx^2)\,g_i
    +(C-\alpha/dx^2\,(g_{x+dx,i}-g_{x-dx,i})=0

We solve this equation by semi-implicit iteration: For each lattice point
:math:`x`, the quadratic equation is solved, assuming the terms
referring to different lattice points known and fixed; this is repeated
for all lattice points until convergence is reached.

The :math:`d`-dimensional lattice is treated with periodic boundary conditions,
and in higher dimensions the finite-difference laplacian is taken over
all unit vectors,

.. math::

    \Delta g\mapsto\frac1{dx^2}\left(
    \sum_{\alpha=1}^d(g_{x+e_\alpha dx}+g_{x-e_\alpha dx})-2dg_{x}\right)

The stencil is currently only implemented for one-, two-, and
three-dimensional lattices.
