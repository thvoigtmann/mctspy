Non-Ergodicity Parameters and Eigenvectors
==========================================

The analysis of the long-time limit of the standard MCT equation leads
to an algebraic equation for the non-ergodicity parameter,

.. math::

    m[f](q) = \frac{f(q)}{1-f(q)}

where :math:`m[f](q)` is the value of the memory kernel evaluated at
the non-erdociity parameter. This non-linear implicit equation has potentially
many roots, and the largest (in some appropriate sense) of it is the
long-time limit of the correlation function. As the parameters entering
the memory kernel are changed, bifurcations appear in this equation,
and these are the MCT glass transition points. The proper analysis of these
bifurcations is via the analysis of the critical eigenvector of the stability
matrix of the equation.

We provide a number of functions to solve for the non-ergodicity parameter,
the associated eigenvalues, and related quantities.

Nonergodicity Parameter
-----------------------

.. autoclass:: mctspy.nonergodicity_parameter
   :members:
   :inherited-members:

Critical Eigenvector and -value
-------------------------------

.. autoclass:: mctspy.eigenvalue
   :members:
   :inherited-members:
