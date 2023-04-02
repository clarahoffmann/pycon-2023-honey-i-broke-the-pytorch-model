" Utilities for discrete variable creation"
# pylint: disable=import-error
from typing import Tuple

import jax
import jax.numpy as jnp
from jax import random

MEANS = jnp.array([[1, 2], [-1, -2], [0, 5]])
VARIANCES = jnp.array(
    [[[1, 0.1], [0.1, 1]], [[1, -0.1], [0.3, 1]], [[1, -0.5], [-0.7, 1]]]
)
KEY = random.PRNGKey(0)
NUM_SAMPLES = 500
NUM_CLASSES = 2

def generate_mv_data(
    key: random.PRNGKey,
    means: jnp.array,
    variances: jnp.array,
    num_samples: int,
    num_labels: int,
) -> Tuple[jnp.array, jnp.array]:
    """
    Generate from multiple multivariate normals at once
    """
    # vectorize multivariate normal distribution
    mvn_func = jax.vmap(random.multivariate_normal, in_axes=(None, 0, 0, None))
    data = mvn_func(key, means, variances, (num_samples,))
    labels = jnp.stack(
        [
            jnp.repeat(jnp.array([i]), num_samples, axis=0)
            for i in range(num_labels)
        ]
    )
    return data, labels