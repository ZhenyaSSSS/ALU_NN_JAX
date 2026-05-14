import jax
import jax.numpy as jnp
import optax
from optax import contrib
from config import config

def tpu_mmd_imq_loss(z: jnp.ndarray, key: jax.Array) -> jnp.ndarray:
    ss = config.MMD_IMQ_SUB_SAMPLE
    step = max(1, z.shape[0] // ss)
    z_sub = z[::step][:ss].astype(jnp.float32)
    
    ref = jax.random.normal(key, z_sub.shape)
    
    def sq_pairwise(x, y):
        x_sq = jnp.sum(jnp.square(x), axis=-1, keepdims=True)
        y_sq = jnp.sum(jnp.square(y), axis=-1, keepdims=True)
        xy = jnp.matmul(x, y.T)
        return jnp.maximum(0.0, x_sq + y_sq.T - 2.0 * xy)
        
    d_xx = sq_pairwise(z_sub, z_sub)
    d_yy = sq_pairwise(ref, ref)
    d_xy = sq_pairwise(z_sub, ref)
    
    scales = jnp.array(config.MMD_IMQ_SCALES, dtype=jnp.float32)
    dim = float(z.shape[-1])
    c = jnp.reshape(scales, (-1, 1, 1)) * dim
    
    def k_mean(dist_sq):
        return jnp.mean(jnp.sum(c / (c + jnp.expand_dims(dist_sq, 0)), axis=0))
        
    mmd2 = k_mean(d_xx) + k_mean(d_yy) - 2.0 * k_mean(d_xy)
    return jnp.maximum(1e-6, mmd2)

def compute_losses(logits_clean, logits_solver, target_bits, pred_z, enc_Target, swd_key):
    logits_clean = logits_clean.astype(jnp.float32)
    logits_solver = logits_solver.astype(jnp.float32)
    target_f32 = target_bits.astype(jnp.float32)
    bce_clean = optax.sigmoid_binary_cross_entropy(logits_clean, target_f32)
    bce_solver = optax.sigmoid_binary_cross_entropy(logits_solver, target_f32)
    
    bit_weights = config.get_bit_weights()
    loss_bce_clean = jnp.mean(bce_clean * bit_weights)
    loss_bce_solver = jnp.mean(bce_solver * bit_weights)
    
    loss_bce = loss_bce_clean + config.BETA_DECODER * loss_bce_solver
    
    loss_latent_reg = jnp.mean(optax.huber_loss(pred_z, jax.lax.stop_gradient(enc_Target), delta=1.0))
    
    loss_wae = tpu_mmd_imq_loss(enc_Target, swd_key)
    
    loss = (
        config.LAMBDAS["bce"] * loss_bce
        + config.LAMBDAS["latent_reg"] * loss_latent_reg
        + config.LAMBDAS["wae"] * loss_wae
    )
    
    metrics = {
        "train/loss": loss,
        "train/loss_total": loss,
        "train/bce": loss_bce,
        "train/bce_clean": loss_bce_clean,
        "train/bce_solver": loss_bce_solver,
        "train/latent_reg": loss_latent_reg,
        "train/wae_mmd": loss_wae,
    }
    
    return loss, metrics

def _schedule_free_lr_schedule():
    """Linear warmup to peak LR, then constant (no cosine decay). Required by schedule-free."""
    return optax.warmup_constant_schedule(
        init_value=0.0,
        peak_value=config.LR,
        warmup_steps=config.SCHEDULE_FREE_WARMUP_STEPS,
    )


def unwrap_schedule_free_state(opt_state) -> object:
    """opt_state may be ScheduleFreeState or a tuple from optax.chain."""
    if hasattr(opt_state, "b1") and hasattr(opt_state, "z"):
        return opt_state
    if isinstance(opt_state, (tuple, list)):
        for s in opt_state:
            if hasattr(s, "b1") and hasattr(s, "z"):
                return s
    raise ValueError("Expected ScheduleFreeState inside optimizer state (enable USE_SCHEDULE_FREE or fix chain).")


def eval_params_for_inference(params, opt_state):
    """Training params vs eval-averaged params for schedule-free."""
    if not getattr(config, "USE_SCHEDULE_FREE", True):
        return params
    sf_state = unwrap_schedule_free_state(opt_state)
    return contrib.schedule_free_eval_params(sf_state, params)


def get_optimizer():
    clip = optax.clip_by_global_norm(config.GRAD_CLIP)
    if not getattr(config, "USE_SCHEDULE_FREE", True):
        total_steps = config.EPOCHS * config.STEPS_PER_EPOCH
        warmup_steps = min(int(config.LR_WARMUP_STEPS), max(1, total_steps - 1))
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=config.LR,
            warmup_steps=warmup_steps,
            decay_steps=total_steps,
            end_value=config.LR * config.LR_COSINE_MIN_RATIO,
        )
        return optax.chain(clip, optax.adamw(learning_rate=schedule, weight_decay=config.WEIGHT_DECAY))

    lr_sched = _schedule_free_lr_schedule()
    b1_sf, b2 = config.SCHEDULE_FREE_BETAS
    wlp = config.SCHEDULE_FREE_WEIGHT_LR_POWER
    variant = str(getattr(config, "SCHEDULE_FREE_VARIANT", "adamw")).lower().strip()

    if variant == "radam":
        # Base optimizer must have no momentum; schedule-free supplies b1_sf averaging.
        inner = contrib.schedule_free(
            optax.chain(
                optax.add_decayed_weights(config.WEIGHT_DECAY),
                optax.radam(learning_rate=lr_sched, b1=0.0, b2=b2, eps=1e-8),
            ),
            learning_rate=lr_sched,
            b1=b1_sf,
            weight_lr_power=wlp,
        )
    else:
        inner = contrib.schedule_free_adamw(
            learning_rate=config.LR,
            warmup_steps=config.SCHEDULE_FREE_WARMUP_STEPS,
            b1=b1_sf,
            b2=b2,
            eps=1e-8,
            weight_decay=config.WEIGHT_DECAY,
            weight_lr_power=wlp,
        )
    return optax.chain(clip, inner)
