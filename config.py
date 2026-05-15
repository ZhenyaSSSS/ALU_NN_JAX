import jax.numpy as jnp
from typing import Dict, Tuple

class Config:
    SEED: int = 42

    WANDB_PROJECT: str = "ALU_NN_JAX"
    WANDB_RUN_NAME: str | None = None
    WANDB_LOG_MODEL: bool = True

    LATENT_DIM: int = 512
    NUM_BITS: int = 32
    NUM_LAYERS_ENC: int = 7
    NUM_LAYERS_REFINE: int = 16
    NUM_DECODER_BLOCKS: int = 2

    COMPUTE_DTYPE: jnp.dtype = jnp.bfloat16

    BATCH_SIZE_PER_DEVICE: int = 65536
    
    LR: float = 2.5e-3
    LR_WARMUP_STEPS: int = 10000
    LR_COSINE_MIN_RATIO: float = 0.01

    USE_SCHEDULE_FREE: bool = True
    SCHEDULE_FREE_VARIANT: str = "adamw"
    SCHEDULE_FREE_WARMUP_STEPS: int = 10000
    SCHEDULE_FREE_BETAS: Tuple[float, float] = (0.9, 0.999)
    SCHEDULE_FREE_R: float = 0.0
    SCHEDULE_FREE_WEIGHT_LR_POWER: float = 2.0
    WEIGHT_DECAY: float = 1e-2
    GRAD_CLIP: float = 1.0
    EPOCHS: int = 350
    STEPS_PER_EPOCH: int = 1000

    NOISE_LEVEL: float = 0.04

    MMD_IMQ_SUB_SAMPLE: int = 1024
    MMD_IMQ_SCALES: Tuple[float, ...] = (0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0)

    BETA_DECODER: float = 1.0

    LAMBDAS: Dict[str, float] = {
        "bce": 1.0, 
        "wae": 0.1, 
        "latent_reg": 0.0
    }

    QUIET_NAN: int = 2143289344

    OPERATIONS_MAP: Dict[str, Tuple[int, bool]] = {
        "add": (0, False),
        "sub": (1, False),
        "mul": (2, False),
        "div": (3, False),
        "sin": (4, True),
        "cos": (5, True),
        "exp": (6, True),
        "log": (7, True),
    }
    NUM_OPS: int = len(OPERATIONS_MAP)

    @staticmethod
    def get_bit_weights():
        weights = jnp.ones(32)
        weights = weights.at[0].set(10.0)
        weights = weights.at[1:10].set(5.0)
        weights = weights.at[10:32].set(jnp.linspace(1.0, 0.3, num=22))
        return weights / weights.mean()

config = Config()
