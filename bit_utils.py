import jax
import jax.numpy as jnp
from config import config

def canonicalize_float(f32_tensor: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    f32_tensor = f32_tensor.astype(jnp.float32)
    int_repr = jax.lax.bitcast_convert_type(f32_tensor, jnp.int32)
    int_repr = jnp.where(int_repr == jnp.iinfo(jnp.int32).min, jnp.zeros_like(int_repr), int_repr)
    f32_tensor = jax.lax.bitcast_convert_type(int_repr, jnp.float32)
    nan_mask = jnp.isnan(f32_tensor)
    return f32_tensor, nan_mask

def float32_to_bits(float_tensor: jnp.ndarray) -> jnp.ndarray:
    f, nan_mask = canonicalize_float(float_tensor)
    f_clean = jnp.where(nan_mask, jnp.zeros_like(f), f)
    xi = jax.lax.bitcast_convert_type(f_clean, jnp.int32)
    quiet = jnp.array(config.QUIET_NAN, dtype=jnp.int32)
    xi = jnp.where(nan_mask, quiet, xi)
    
    shifts = jnp.arange(31, -1, -1, dtype=jnp.int32)
    bits = jnp.bitwise_and(jnp.right_shift(jnp.expand_dims(xi, -1), shifts), 1)
    return bits.astype(jnp.int32)

def bits_to_float(bits_tensor: jnp.ndarray) -> jnp.ndarray:
    bits = bits_tensor.astype(jnp.int32)
    shifts = jnp.arange(31, -1, -1, dtype=jnp.int32)
    shifted = jnp.left_shift(bits, shifts)
    xi = jnp.sum(shifted, axis=-1, dtype=jnp.int32)
    f, _ = canonicalize_float(jax.lax.bitcast_convert_type(xi, jnp.float32))
    return f
