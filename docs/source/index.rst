.. ZhMaR documentation master file, created by
   sphinx-quickstart on Sat May 12 12:56:07 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. module:: sphinx.ext.mathbase
   :synopsis: Common math support for pngmath and jsmath.


ZhMaR
=================================

    ZhMaR is a ```PyTorch```-based wrapper to conduct Stochastical Variational Inference with an arbitary probabilistic model.

.. toctree::
   :maxdepth: 2
    license

   :caption: Contents:

Variational Inference
=====================


    .. automodule:: BBSVI
        :members:
        :undoc-members:
        :show-inheritance:

    Consider hierarchical probabilistic model 

        .. math::

            p_{\theta}(x_i, z_i, \beta) = p_{\theta}(x_i | z_i, \beta) p_\theta(z_i | \beta) p_{\theta}(\beta),

    where :math:`x_i` is an observed variable, :math:`z_i` is a corresponding local latent variable, :math:`\beta` is a global latent variable and :math:`\theta` is a parameter of our model.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
