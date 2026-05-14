import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Any


class RMSNorm(nn.Module):
    eps: float = 1e-6

    @nn.compact
    def __call__(self, x):
        norm = jax.lax.rsqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + self.eps)
        weight = self.param("weight", nn.initializers.ones, (x.shape[-1],))
        return x * norm * weight


class GeGLU(nn.Module):
    dim_out: int
    dtype: Any = jnp.bfloat16

    @nn.compact
    def __call__(self, x):
        proj = nn.Dense(self.dim_out * 2, dtype=self.dtype)(x)
        x, gate = jnp.split(proj, 2, axis=-1)
        return x * nn.gelu(gate)


class ResMLPBlock(nn.Module):
    dim: int
    dtype: Any = jnp.bfloat16

    @nn.compact
    def __call__(self, x):
        residual = x
        x = RMSNorm()(x)
        x = GeGLU(dim_out=self.dim, dtype=self.dtype)(x)
        return residual + x


_RematResMLPBlock = nn.remat(ResMLPBlock)


class BitEncoder(nn.Module):
    latent_dim: int
    num_layers: int
    num_bits: int = 32
    dtype: Any = jnp.bfloat16

    @nn.compact
    def __call__(self, bits):
        embed_dim = self.latent_dim // self.num_bits
        x = nn.Embed(num_embeddings=2, features=embed_dim, dtype=self.dtype)(bits)
        x = x.reshape((x.shape[0], -1))
        for _ in range(self.num_layers):
            x = _RematResMLPBlock(dim=self.latent_dim, dtype=self.dtype)(x)
        return x


class LatentSolver(nn.Module):
    num_ops: int
    latent_dim: int
    num_refine: int
    dtype: Any = jnp.bfloat16

    @nn.compact
    def __call__(self, A: jnp.ndarray, B: jnp.ndarray, op_ids: jnp.ndarray):
        op_embed = nn.Embed(num_embeddings=self.num_ops, features=self.latent_dim, dtype=self.dtype)(op_ids)
        x = jnp.concatenate([A, B, op_embed, A * B], axis=-1)
        x = nn.Dense(self.latent_dim, dtype=self.dtype)(x)
        for _ in range(self.num_refine):
            x = _RematResMLPBlock(dim=self.latent_dim, dtype=self.dtype)(x)
        return x


class BitDecoder(nn.Module):
    latent_dim: int
    num_blocks: int
    dtype: Any = jnp.bfloat16

    @nn.compact
    def __call__(self, z):
        for _ in range(self.num_blocks):
            z = _RematResMLPBlock(dim=self.latent_dim, dtype=self.dtype)(z)
        z = RMSNorm()(z)
        return nn.Dense(32, dtype=self.dtype)(z)


class NLA(nn.Module):
    config: Any

    def setup(self):
        dt = self.config.COMPUTE_DTYPE
        self.encoder = BitEncoder(
            latent_dim=self.config.LATENT_DIM,
            num_layers=self.config.NUM_LAYERS_ENC,
            num_bits=self.config.NUM_BITS,
            dtype=dt,
        )
        self.solver = LatentSolver(
            num_ops=self.config.NUM_OPS,
            latent_dim=self.config.LATENT_DIM,
            num_refine=self.config.NUM_LAYERS_REFINE,
            dtype=dt,
        )
        self.decoder = BitDecoder(
            latent_dim=self.config.LATENT_DIM,
            num_blocks=self.config.NUM_DECODER_BLOCKS,
            dtype=dt,
        )

    def __call__(self, bits_A, bits_B, op_ids):
        enc_A = self.encoder(bits_A)
        enc_B = self.encoder(bits_B)
        z = self.solver(enc_A, enc_B, op_ids)
        return self.decoder(z)
