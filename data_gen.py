import jax
import jax.numpy as jnp
import math
from config import config
from bit_utils import float32_to_bits

_ORDERED_OPS = sorted(config.OPERATIONS_MAP.items(), key=lambda kv: kv[1][0])
_OP_IDS = [oid for _, (oid, _) in _ORDERED_OPS]
assert _OP_IDS == list(range(len(_OP_IDS))), "OPERATIONS_MAP op ids must be 0..N-1"

def generate_batch(key: jax.Array, batch_size: int):
    num_ops = len(_ORDERED_OPS)

    n_uni = int(batch_size * 0.3)
    n_log = int(batch_size * 0.3)
    n_int = int(batch_size * 0.15)
    n_mic = int(batch_size * 0.05)
    n_edg = int(batch_size * 0.1)
    n_trap = batch_size - (n_uni + n_log + n_int + n_mic + n_edg)

    keys = jax.random.split(key, 13)

    A_uni = jax.random.uniform(keys[0], (n_uni,), minval=-1e5, maxval=1e5)
    B_uni = jax.random.uniform(keys[1], (n_uni,), minval=-1e5, maxval=1e5)

    s_la = jax.random.normal(keys[2], (n_log,))
    e_la = jax.random.uniform(keys[3], (n_log,), minval=-30, maxval=30)
    A_log = jnp.sign(s_la) * (10 ** e_la)
    
    s_lb = jax.random.normal(keys[4], (n_log,))
    e_lb = jax.random.uniform(keys[5], (n_log,), minval=-30, maxval=30)
    B_log = jnp.sign(s_lb) * (10 ** e_lb)

    A_int = jax.random.randint(keys[6], (n_int,), minval=-1000, maxval=1000).astype(jnp.float32)
    B_int = jax.random.randint(keys[7], (n_int,), minval=-1000, maxval=1000).astype(jnp.float32)

    A_mic = jax.random.normal(keys[8], (n_mic,))
    B_mic = A_mic + 1e-5 * jax.random.normal(keys[9], (n_mic,))

    edges = jnp.array([0.0, -0.0, 1.0, -1.0, math.pi, math.e], dtype=jnp.float32)
    A_edg = edges[jax.random.randint(keys[10], (n_edg,), 0, len(edges))]
    B_edg = edges[jax.random.randint(keys[11], (n_edg,), 0, len(edges))]

    keys_trap = jax.random.split(keys[12], 2)
    A_trap = jax.random.normal(keys_trap[0], (n_trap,)) * 1e30
    B_trap = jnp.zeros((n_trap,), dtype=jnp.float32)

    A = jnp.concatenate([A_uni, A_log, A_int, A_mic, A_edg, A_trap], axis=0)
    B = jnp.concatenate([B_uni, B_log, B_int, B_mic, B_edg, B_trap], axis=0)

    op_idx = jax.random.randint(keys_trap[1], (batch_size,), 0, num_ops)

    parts = []
    for name, _ in _ORDERED_OPS:
        if name == "add":
            parts.append(A + B)
        elif name == "sub":
            parts.append(A - B)
        elif name == "mul":
            parts.append(A * B)
        elif name == "div":
            parts.append(A / B)
        elif name == "sin":
            parts.append(jnp.sin(A))
        elif name == "cos":
            parts.append(jnp.cos(A))
        elif name == "exp":
            parts.append(jnp.exp(A))
        elif name == "log":
            parts.append(jnp.log(A))
        else:
            raise RuntimeError(f"unknown op {name}")
            
    stacked = jnp.stack(parts, axis=1)
    Target = jnp.take_along_axis(stacked, jnp.expand_dims(op_idx, axis=1), axis=1).squeeze(axis=1)

    bits_A = float32_to_bits(A)
    bits_B = float32_to_bits(B)
    bits_Target = float32_to_bits(Target)

    return bits_A, bits_B, op_idx, bits_Target
