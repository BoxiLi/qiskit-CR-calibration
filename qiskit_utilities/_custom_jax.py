# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file has been modified to add support for jvp with auxiliary variables.

# https://github.com/google/jax/commit/d05431a1ffd27abc3951d723c82fa63db88bdd69

from functools import partial

import jax
from jax import vmap
from jax.tree_util import tree_map
from jax._src.api_util import (
    argnums_partial,
)
from jax._src.api import (
    _std_basis,
    _jacfwd_unravel,
    _jvp,
)

try:
    from jax.extend.linear_util import wrap_init
except:
    from jax.linear_util import wrap_init


def _value_and_jacfwd(fun, argnums=0, has_aux=False):
    """
    A custom function that computes the value of a function and the Jacobian
    using forward methods.
    Officially JAX only supports value_and_jacrev.
    For the forward method, saving the function value only reduces the time by a factor of 2.
    So they are not very interested in this.
    However, the forward method is more useful for us because the operator (which is a matrix) depends on a single parameter, the time.
    This also adds support for auxiliary values.
    """

    def jacfun(*args, **kwargs):
        f = wrap_init(fun, kwargs)
        f_partial, dyn_args = argnums_partial(
            f, argnums, args, require_static_args_hashable=False
        )
        # tree_map(partial(_check_input_dtype_jacfwd, holomorphic), dyn_args)
        if not has_aux:
            pushfwd = partial(_jvp, f_partial, dyn_args)
            y, jac = vmap(pushfwd, out_axes=(None, -1))(_std_basis(dyn_args))
        else:
            pushfwd = partial(_jvp, f_partial, dyn_args, has_aux=True)
            y, jac, aux = vmap(pushfwd, out_axes=(None, -1, None))(_std_basis(dyn_args))
        # tree_map(partial(_check_output_dtype_jacfwd, holomorphic), y)
        example_args = dyn_args[0] if isinstance(argnums, int) else dyn_args
        jac_tree = tree_map(partial(_jacfwd_unravel, example_args), y, jac)
        if not has_aux:
            return y, jac_tree
        else:
            return (y, aux), jac_tree

    return jacfun
