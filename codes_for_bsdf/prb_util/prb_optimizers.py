from contextlib import contextmanager
from collections import defaultdict
import drjit as dr
import mitsuba as mi
import numpy as np
import torch
import time

from custom_ops.filter import AtrousFilter, Atrous1DFilter, GaussianFilter, Gaussian1DFilter
from custom_ops.james_stein import JS_opt


    
class Optimizer:
    """
    Base class of all gradient-based optimizers.
    """
    def __init__(self, lr, params: dict):
        """
        Parameter ``lr``:
            learning rate

        Parameter ``params`` (:py:class:`dict`):
            Dictionary-like object containing parameters to optimize.
        """
        self.lr   = defaultdict(lambda: self.lr_default)
        self.lr_v = defaultdict(lambda: self.lr_default_v)

        self.set_learning_rate(lr)
        self.variables = {}
        self.state = {}
  
        self.biased_state = {}
        self.prev_grad ={}
        # self.unbiased_variables = {}
        self.biased_variables ={}

        if params is not None:
            for k, v in params.items():
                if type(params) is mi.SceneParameters:
                    if params.flags(k) & mi.ParamFlags.NonDifferentiable.value:
                        continue
                self.__setitem__(k, v)

    def __contains__(self, key: str):
        return self.variables.__contains__(key)

    def __getitem__(self, key: str):
        return self.variables[key]

    def __setitem__(self, key: str, value):
        """
        Overwrite the value of a parameter.

        This method can also be used to load a scene parameters to optimize.

        Note that the state of the optimizer (e.g. momentum) associated with the
        parameter is preserved, unless the parameter's dimensions changed. When
        assigning a substantially different parameter value (e.g. as part of a
        different optimization run), the previous momentum value is meaningless
        and calling `reset()` is advisable.
        """
        if not (dr.is_diff_v(value) and dr.is_float_v(value)):
            raise Exception('Optimizer.__setitem__(): value should be differentiable!')
        needs_reset = (key not in self.variables) or dr.shape(self.variables[key]) != dr.shape(value)

        self.variables[key] = dr.detach(value, True)
        dr.enable_grad(self.variables[key])
        if needs_reset:
            self.reset(key)

    def __delitem__(self, key: str) -> None:
        del self.variables[key]

    def __len__(self) -> int:
        return len(self.variables)

    def keys(self):
        return self.variables.keys()

    def items(self):
        class OptimizerItemIterator:
            def __init__(self, items):
                self.items = items
                self.it = items.keys().__iter__()

            def __iter__(self):
                return self

            def __next__(self):
                key = next(self.it)
                return (key, self.items[key])

        return OptimizerItemIterator(self.variables)

    def set_learning_rate(self, lr) -> None:
        """
        Set the learning rate.

        Parameter ``lr`` (``float``, ``dict``):
            The new learning rate. A ``dict`` can be provided instead to
            specify the learning rate for specific parameters.
        """

        # We use `dr.opaque` so the that the JIT compiler does not include
        # the learning rate as a scalar literal into generated code, which
        # would defeat kernel caching when updating learning rates.
        if isinstance(lr, float) or isinstance(lr, int):
            self.lr_default = float(lr)
            self.lr_default_v = dr.opaque(dr.detached_t(mi.Float), lr, shape=1)
        elif isinstance(lr, dict):
            for k, v in lr.items():
                self.lr[k] = v
                self.lr_v[k] = dr.opaque(dr.detached_t(mi.Float), v, shape=1)
        else:
            raise Exception('Optimizer.set_learning_rate(): value should be a float or a dict!')

    def reset(self, key):
        """
        Resets the internal state associated with a parameter, if any (e.g. momentum).
        """
        pass


class SGD(Optimizer):
    """
    Implements basic stochastic gradient descent with a fixed learning rate
    and, optionally, momentum :cite:`Sutskever2013Importance` (0.9 is a typical
    parameter value for the ``momentum`` parameter).

    The momentum-based SGD uses the update equation

    .. math::

        v_{i+1} = \\mu \\cdot v_i +  g_{i+1}

    .. math::
        p_{i+1} = p_i + \\varepsilon \\cdot v_{i+1},

    where :math:`v` is the velocity, :math:`p` are the positions,
    :math:`\\varepsilon` is the learning rate, and :math:`\\mu` is
    the momentum parameter.
    """
    def __init__(self, lr, momentum=0, mask_updates=False, params:dict=None):
        """
        Parameter ``lr``:
            learning rate

        Parameter ``momentum``:
            momentum factor

        Parameter ``mask_updates``:
            if enabled, parameters and state variables will only be updated
            in a given iteration if it received nonzero gradients in that iteration.
            This only has an effect if momentum is enabled.
            See :py:class:`mitsuba.optimizers.Adam`'s documentation for more details.

        Parameter ``params`` (:py:class:`dict`):
            Optional dictionary-like object containing parameters to optimize.
        """
        assert momentum >= 0 and momentum < 1
        assert lr > 0
        self.momentum = momentum
        self.mask_updates = mask_updates
        super().__init__(lr, params)

    def step(self):
        """Take a gradient step"""
        for k, p in self.variables.items():
            g_p = dr.grad(p)
            shape = dr.shape(g_p)
            if shape == 0:
                continue

            if self.momentum != 0:
                if shape != dr.shape(self.state[k]):
                    # Reset state if data size has changed
                    self.reset(k)

                next_state = self.momentum * self.state[k] + g_p
                step = self.lr_v[k] * self.state[k]
                if self.mask_updates:
                    nonzero = dr.neq(g_p, 0.)
                    next_state = dr.select(nonzero, next_state, self.state[k])
                    step = dr.select(nonzero, step, 0)
                self.state[k] = next_state
                value = dr.detach(p) - step
                dr.schedule(self.state[k])
            else:
                value = dr.detach(p) - self.lr_v[k] * g_p


            value = type(p)(value)
            dr.enable_grad(value)
            self.variables[k] = value
            dr.schedule(self.variables[k])

        dr.eval()

    def reset(self, key):
        """Zero-initializes the internal state associated with a parameter"""
        if self.momentum == 0:
            return
        p = self.variables[key]
        shape = dr.shape(p) if p.IsTensor else dr.width(p)
        self.state[key] = dr.zeros(dr.detached_t(p), shape)

    def __repr__(self):
        return ('SGD[\n'
                '  variables = %s,\n'
                '  lr = %s,\n'
                '  momentum = %.2g\n'
                ']' % (list(self.keys()), dict(self.lr, default=self.lr_default),
                       self.momentum))


class CustomAdam(Optimizer):
    """
    Implements the Adam optimizer presented in the paper *Adam: A Method for
    Stochastic Optimization* by Kingman and Ba, ICLR 2015.

    When optimizing many variables (e.g. a high resolution texture) with
    momentum enabled, it may be beneficial to restrict state and variable
    updates to the entries that received nonzero gradients in the current
    iteration (``mask_updates=True``).
    In the context of differentiable Monte Carlo simulations, many of those
    variables may not be observed at each iteration, e.g. when a surface is
    not visible from the current camera. Gradients for unobserved variables
    will remain at zero by default.
    If we do not take special care, at each new iteration:

    1. Momentum accumulated at previous iterations (potentially very noisy)
       will keep being applied to the variable.
    2. The optimizer's state will be updated to incorporate ``gradient = 0``,
       even though it is not an actual gradient value but rather lack of one.

    Enabling ``mask_updates`` avoids these two issues. This is similar to
    `PyTorch's SparseAdam optimizer <https://pytorch.org/docs/1.9.0/generated/torch.optim.SparseAdam.html>`_.
    """
    def __init__(self, lr, beta_1=0.9, beta_2=0.999, epsilon=1e-8,
                 mask_updates=False, uniform=False, params: dict=None):
        """
        Parameter ``lr``:
            learning rate

        Parameter ``beta_1``:
            controls the exponential averaging of first order gradient moments

        Parameter ``beta_2``:
            controls the exponential averaging of second order gradient moments

        Parameter ``mask_updates``:
            if enabled, parameters and state variables will only be updated in a
            given iteration if it received nonzero gradients in that iteration

        Parameter ``uniform``:
            if enabled, the optimizer will use the 'UniformAdam' variant of Adam
            [Nicolet et al. 2021], where the update rule uses the *maximum* of
            the second moment estimates at the current step instead of the
            per-element second moments.

        Parameter ``params`` (:py:class:`dict`):
            Optional dictionary-like object containing parameters to optimize.
        """
        assert 0 <= beta_1 < 1 and 0 <= beta_2 < 1 \
            and lr > 0 and epsilon > 0

        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.mask_updates = mask_updates
        self.uniform = uniform
        self.t = defaultdict(lambda: 0)
        super().__init__(lr, params)

    
    def step(self):
        """Take a gradient step"""

        for k, p in self.variables.items():
            self.t[k] += 1
            lr_scale = dr.sqrt(1 - self.beta_2 ** self.t[k]) / (1 - self.beta_1 ** self.t[k])
            lr_scale = dr.opaque(dr.detached_t(mi.Float), lr_scale, shape=1)

            lr_t = self.lr_v[k] * lr_scale
            g_p = dr.grad(p)
            shape = dr.shape(g_p)
            
            if shape == 0:
                continue
            elif shape != dr.shape(self.state[k][0]):
                # Reset state if data size has changed
                self.reset(k)


            m_tp, v_tp = self.state[k]
            
            m_t = self.beta_1 * m_tp + (1 - self.beta_1) * g_p
            v_t = self.beta_2 * v_tp + (1 - self.beta_2) * dr.sqr(g_p)

            if self.mask_updates:
                nonzero = dr.neq(g_p, 0.)
                m_t = dr.select(nonzero, m_t, m_tp)
                v_t = dr.select(nonzero, v_t, v_tp)
            self.state[k] = (m_t, v_t)
            dr.schedule(self.state[k])

            if self.uniform:
                step = lr_t * m_t / (dr.sqrt(dr.max(v_t)) + self.epsilon)
            else:
                step = lr_t * m_t / (dr.sqrt(v_t) + self.epsilon)
            if self.mask_updates:
                step = dr.select(nonzero, step, 0.)
            u = dr.detach(p) - step

            u = type(p)(u)
            dr.enable_grad(u)
            self.variables[k] = u
            dr.schedule(self.variables[k])

        dr.eval()

    def reset(self, key):
        """Zero-initializes the internal state associated with a parameter"""
        p = self.variables[key]
        shape = dr.shape(p) if p.IsTensor else dr.width(p)
        self.state[key] = (dr.zeros(dr.detached_t(p), shape),
                           dr.zeros(dr.detached_t(p), shape))
        self.t[key] = 0

    def __repr__(self):
        return ('Adam[\n'
                '  variables = %s,\n'
                '  lr = %s,\n'
                '  betas = (%g, %g),\n'
                '  eps = %g\n'
                ']' % (list(self.keys()), dict(self.lr, default=self.lr_default),
                       self.beta_1, self.beta_2, self.epsilon))

class CustomAdamJS(Optimizer):
    """
    Implements the Adam optimizer presented in the paper *Adam: A Method for
    Stochastic Optimization* by Kingman and Ba, ICLR 2015.

    When optimizing many variables (e.g. a high resolution texture) with
    momentum enabled, it may be beneficial to restrict state and variable
    updates to the entries that received nonzero gradients in the current
    iteration (``mask_updates=True``).
    In the context of differentiable Monte Carlo simulations, many of those
    variables may not be observed at each iteration, e.g. when a surface is
    not visible from the current camera. Gradients for unobserved variables
    will remain at zero by default.
    If we do not take special care, at each new iteration:

    1. Momentum accumulated at previous iterations (potentially very noisy)
       will keep being applied to the variable.
    2. The optimizer's state will be updated to incorporate ``gradient = 0``,
       even though it is not an actual gradient value but rather lack of one.

    Enabling ``mask_updates`` avoids these two issues. This is similar to
    `PyTorch's SparseAdam optimizer <https://pytorch.org/docs/1.9.0/generated/torch.optim.SparseAdam.html>`_.
    """
    def __init__(self, lr, beta_1=0.9, beta_2=0.999, epsilon=1e-8,
                 mask_updates=False, uniform=False, use_js=False, use_gaussian=False, use_cb=False, params: dict=None):
        """
        Parameter ``lr``:
            learning rate

        Parameter ``beta_1``:
            controls the exponential averaging of first order gradient moments

        Parameter ``beta_2``:
            controls the exponential averaging of second order gradient moments

        Parameter ``mask_updates``:
            if enabled, parameters and state variables will only be updated in a
            given iteration if it received nonzero gradients in that iteration

        Parameter ``uniform``:
            if enabled, the optimizer will use the 'UniformAdam' variant of Adam
            [Nicolet et al. 2021], where the update rule uses the *maximum* of
            the second moment estimates at the current step instead of the
            per-element second moments.

        Parameter ``params`` (:py:class:`dict`):
            Optional dictionary-like object containing parameters to optimize.
        """
        assert 0 <= beta_1 < 1 and 0 <= beta_2 < 1 \
            and lr > 0 and epsilon > 0
        
        self.cross_bilateral_op = AtrousFilter(numIter=4)
        self.cross_bilateral_1d_op = Atrous1DFilter(numIter=4)
        self.gaussian_op = GaussianFilter(winSize=31)
        self.gaussian_1d_op = Gaussian1DFilter(winSize=31)

        self.mask_updates = mask_updates
        self.mask_update = 0
        if self.mask_updates:
            self.mask_update = 1
        
        self.js_opt_op = JS_opt(win_size=7)
        self.tilde_g = 0

        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.alpha = 0.9
        self.use_js = use_js
        self.use_cb = use_cb
        self.use_gaussian = use_gaussian
        self.uniform = uniform
        self.t = defaultdict(lambda: 0)
        super().__init__(lr, params)


    @dr.wrap_ad(source='drjit', target='torch')
    def james_stein_combiner_opt(self, unbiased_g, biased_g, g_var, param):
        out_g, rho = self.js_opt_op(unbiased_g.cuda(), biased_g.cuda(), g_var.cuda(), param.cuda())
        return out_g, rho
    
    @dr.wrap_ad(source='drjit', target='torch')
    def cross_bilateral(self, g_p, param):
        out_g_p = self.cross_bilateral_op(g_p, param)
        return out_g_p
    
    @dr.wrap_ad(source='drjit', target='torch')
    def cross_bilateral_1d(self, g_p, param):
        out_g_p = self.cross_bilateral_1d_op(g_p, param)
        return out_g_p

    @dr.wrap_ad(source='drjit', target='torch')
    def gaussian_filter(self, g_p):
        out_g_p = self.gaussian_op(g_p)
        return out_g_p
    
    @dr.wrap_ad(source='drjit', target='torch')
    def gaussian_filter_1d(self, g_p):
        out_g_p = self.gaussian_1d_op(g_p)
        return out_g_p
    
    def get_tilde_g(self, tilde_g):
        self.tilde_g = tilde_g


    def step(self):
        """Take a gradient step"""

        for k, p in self.variables.items():
            self.t[k] += 1

            lr_scale = dr.sqrt(1 - self.beta_2 ** self.t[k]) / (1 - self.beta_1 ** self.t[k])
            lr_scale = dr.opaque(dr.detached_t(mi.Float), lr_scale, shape=1)

            lr_t = self.lr_v[k] * lr_scale
            g_p = dr.grad(p)

            shape = dr.shape(g_p)
        
            if shape == 0:
                continue
            elif shape != dr.shape(self.state[k][0]):
                # Reset state if data size has changed
                self.reset(k)
            
  
            """
             First and second moment estimation for unbiased gradient estimates
            """   
            m_tp, v_tp = self.state[k]
            m_t = self.alpha * m_tp + (1 - self.alpha) * g_p
            v_t = self.beta_2 * v_tp + (1 - self.beta_2) * dr.sqr(g_p)

            """
             operators for biased gradients
            """                    
            if self.use_cb:
                #prefilter 
                if shape[2] == 3:
                    biased_g_p = self.cross_bilateral(g_p, dr.detach(p))
                else:
                    biased_g_p = self.cross_bilateral_1d(g_p, dr.detach(p))
            elif self.use_gaussian:
                if shape[2] == 3:
                    biased_g_p = self.gaussian_filter(g_p)
                else:
                    biased_g_p = self.gaussian_filter_1d(g_p)

        
            """
             combine the unbiased and biased gradients using JS combiner
             and update parameteters with combined gradient estiamtes
            """ 
            prev_g = self.prev_grad[k]

            if (self.t[k] == 1):
                mean = m_tp
            else:
                mean = m_tp / (1 - self.alpha**(self.t[k]-1))


            dual_v = dr.sqr(dr.detach(g_p) - prev_g) * 0.5
   
            biased_m_tp, biased_v_tp = self.biased_state[k] 
            if (self.use_js):
                combined_g_p, rho = self.james_stein_combiner_opt(g_p, biased_g_p, dual_v, mean)     
                biased_m_t = self.beta_1 * biased_m_tp + (1 - self.beta_1) * combined_g_p
                biased_v_t = self.beta_2 * biased_v_tp + (1 - self.beta_2) * dr.sqr(combined_g_p)

            else:
                biased_m_t = self.beta_1 * biased_m_tp + (1 - self.beta_1) * biased_g_p
                biased_v_t = self.beta_2 * biased_v_tp + (1 - self.beta_2) * dr.sqr(biased_g_p)     
            
            self.biased_state[k] = (biased_m_t, biased_v_t)
            dr.schedule(self.biased_state[k])

            """
             update mean and prev_g
            """
            self.state[k] = (m_t, v_t)
            dr.schedule(self.state[k])


            self.prev_grad[k] = g_p
            dr.schedule(self.prev_grad[k])

            step = lr_t * biased_m_t / (dr.sqrt(biased_v_t) + self.epsilon)

            u = dr.detach(p) - step
            u = type(p)(u)
            dr.enable_grad(u)
            self.variables[k] = u
            dr.schedule(self.variables[k])

        dr.eval()

    def reset(self, key):
        """Zero-initializes the internal state associated with a parameter"""
        p = self.variables[key]
        shape = dr.shape(p) if p.IsTensor else dr.width(p)
        self.state[key] = (dr.zeros(dr.detached_t(p), shape),
                           dr.zeros(dr.detached_t(p), shape))
        self.biased_state[key] = (dr.zeros(dr.detached_t(p), shape),
                       dr.zeros(dr.detached_t(p), shape))
        
        self.prev_grad[key] = dr.zeros(dr.detached_t(p), shape)
        self.t[key] = 0


    def __repr__(self):
        return ('Adam[\n'
                '  variables = %s,\n'
                '  lr = %s,\n'
                '  betas = (%g, %g),\n'
                '  eps = %g\n'
                ']' % (list(self.keys()), dict(self.lr, default=self.lr_default),
                       self.beta_1, self.beta_2, self.epsilon))


