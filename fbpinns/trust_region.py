from collections.abc import Callable
import functools
from typing import Any, Optional, Union, NamedTuple
from IPython import embed

import jax
import jax.numpy as jnp
# from dataclasses import dataclass, replace
from jax.tree_util import tree_map, tree_leaves
from optax._src import base
from optax._src import clipping
from optax._src import combine
from optax._src import factorized
from optax._src import linesearch as _linesearch
from optax._src import transform
from optax._src import wrappers

MaskOrFn = Optional[Union[Any, Callable[[base.Params], Any]]]

class TrustRegionState(NamedTuple):
    learning_rate: base.ScalarOrSchedule  # PyTree matching the shape of params

class TrustRegion:
    def __init__(self,
                learning_rate: base.ScalarOrSchedule,
                max_learning_rate: float = 1.0,
                min_learning_rate: float = 1e-6,
                max_iterations: int = 1,
                inc_factor: float = 2.0,
                dec_factor: float = 0.5, 
                nu: float = 0.5,
                nu1: float = 0.25,
                nu2: float = 0.75,
                norm_type: int = 2) -> base.GradientTransformation:
        self.initial_learning_rate = learning_rate
        self.max_learning_rate = max_learning_rate
        self.min_learning_rate = min_learning_rate
        self.max_iterations = max_iterations
        self.inc_factor = inc_factor
        self.dec_factor = dec_factor
        self.nu = nu
        self.nu1 = nu1
        self.nu2 = nu2
        self.norm_type = norm_type

    def init(self, params: base.Params) -> TrustRegionState:
        """Initialize the state of the optimizer."""
        return TrustRegionState(
            learning_rate=jax.tree_util.tree_map(lambda p: jnp.full_like(p, self.initial_learning_rate), params)
        )
        
    def update(self, grads, state, params):
        """
        Trust region update with adaptive learning rate.
        """
        # Compute global gradient norm
        grad_norm_sq = sum(jnp.sum(jnp.square(g)) for g in tree_leaves(grads))
        grad_norm = jnp.sqrt(grad_norm_sq)

        # Compute full step norm (scaled by current learning rate)
        full_step_norm_sq = sum(
            jnp.sum(jnp.square(lr * g)) for g, lr in zip(tree_leaves(grads), tree_leaves(state.learning_rate))
        )
        full_step_norm = jnp.sqrt(full_step_norm_sq)

        # Trust region scaling
        scaling = jnp.minimum(1.0, 1.0 / (full_step_norm + 1e-8))

        # Compute updates
        updates = tree_map(lambda g, lr: -lr * g * scaling, grads, state.learning_rate)

        # Dummy ratio for example; replace with actual ratio computation
        ratio = 0.6

        # Update learning rate per parameter
        def update_lr(lr):
            lr = jnp.where(ratio < self.nu1, lr * self.dec_factor, lr)
            lr = jnp.where(ratio > self.nu2, lr * self.inc_factor, lr)
            return jnp.clip(lr, self.min_learning_rate, self.max_learning_rate)

        new_learning_rate = tree_map(update_lr, state.learning_rate)

        # Construct new state directly
        new_state = TrustRegionState(learning_rate=new_learning_rate)

        return updates, new_state

             
def trust_region(**kwargs):
    return TrustRegion(**kwargs)