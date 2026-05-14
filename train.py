import jax
import jax.numpy as jnp
import flax
from flax.training import train_state
import wandb
from tqdm import tqdm
from functools import partial

from config import config
from architecture import NLA, BitEncoder, LatentSolver, BitDecoder
from data_gen import generate_batch
from losses import compute_losses, eval_params_for_inference, get_optimizer, unwrap_schedule_free_state
from bit_utils import bits_to_float

class TrainState(train_state.TrainState):
    pass

def create_train_state(rng):
    model = NLA(config=config)
    
    dummy_bits_A = jnp.zeros((1, 32), dtype=jnp.int32)
    dummy_bits_B = jnp.zeros((1, 32), dtype=jnp.int32)
    dummy_op_ids = jnp.zeros((1,), dtype=jnp.int32)
    
    variables = model.init(rng, dummy_bits_A, dummy_bits_B, dummy_op_ids)
    
    tx = get_optimizer()
    return TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        tx=tx,
    )

@partial(jax.pmap, axis_name='batch')
def train_step(state, key):
    batch_size_per_device = config.BATCH_SIZE_PER_DEVICE
    
    keys = jax.random.split(key, 4)
    data_key, noise_key_a, noise_key_b, swd_key = keys[0], keys[1], keys[2], keys[3]
    
    bits_A, bits_B, op_ids, bits_Target = generate_batch(data_key, batch_size_per_device)
    
    def loss_fn(params):
        dt = config.COMPUTE_DTYPE
        encoder = BitEncoder(
            latent_dim=config.LATENT_DIM,
            num_layers=config.NUM_LAYERS_ENC,
            num_bits=config.NUM_BITS,
            dtype=dt,
        )
        solver = LatentSolver(
            num_ops=config.NUM_OPS,
            latent_dim=config.LATENT_DIM,
            num_refine=config.NUM_LAYERS_REFINE,
            dtype=dt,
        )
        decoder = BitDecoder(
            latent_dim=config.LATENT_DIM,
            num_blocks=config.NUM_DECODER_BLOCKS,
            dtype=dt,
        )
        enc_A = encoder.apply({"params": params["encoder"]}, bits_A)
        enc_B = encoder.apply({"params": params["encoder"]}, bits_B)
        enc_Target = encoder.apply({"params": params["encoder"]}, bits_Target)
        noise_a = jax.random.normal(noise_key_a, enc_A.shape) * config.NOISE_LEVEL * jnp.std(enc_A, axis=-1, keepdims=True)
        noise_b = jax.random.normal(noise_key_b, enc_B.shape) * config.NOISE_LEVEL * jnp.std(enc_B, axis=-1, keepdims=True)
        enc_A_noisy = enc_A + noise_a
        enc_B_noisy = enc_B + noise_b
        pred_z = solver.apply({"params": params["solver"]}, enc_A_noisy, enc_B_noisy, op_ids)
        logits_clean = decoder.apply({"params": params["decoder"]}, enc_Target)
        logits_solver = decoder.apply({"params": params["decoder"]}, pred_z)
        
        loss, metrics = compute_losses(
            logits_clean, logits_solver, bits_Target, 
            pred_z, enc_Target, swd_key
        )
        return loss, metrics
        
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, metrics), grads = grad_fn(state.params)
    
    grads = jax.lax.pmean(grads, axis_name='batch')
    metrics = jax.lax.pmean(metrics, axis_name='batch')
    
    state = state.apply_gradients(grads=grads)
    if getattr(config, "USE_SCHEDULE_FREE", True):
        sf_state = unwrap_schedule_free_state(state.opt_state)
        lr_log = jax.lax.pmean(sf_state.max_lr, axis_name="batch")
        metrics = {**metrics, "train/lr": lr_log}
    else:
        metrics = {**metrics, "train/lr": jnp.array(config.LR, dtype=jnp.float32)}
    return state, metrics

@partial(jax.pmap, axis_name='batch')
def eval_step(state, key):
    batch_size_per_device = config.BATCH_SIZE_PER_DEVICE
    bits_A, bits_B, op_ids, bits_Target = generate_batch(key, batch_size_per_device)

    params_eval = eval_params_for_inference(state.params, state.opt_state)
    logits = state.apply_fn({'params': params_eval}, bits_A, bits_B, op_ids)
    pred_bits = (logits > 0).astype(jnp.int32)
    
    is_correct = jnp.all(pred_bits == bits_Target, axis=-1).astype(jnp.float32)
    ema = jnp.mean(is_correct)
    
    target_f = bits_to_float(bits_Target)
    pred_f = bits_to_float(pred_bits)
    
    nan_target = jnp.isnan(target_f)
    nan_pred = jnp.isnan(pred_f)
    nan_acc = jnp.mean((nan_target == nan_pred).astype(jnp.float32))
    
    fin = jnp.isfinite(target_f) & jnp.isfinite(pred_f)
    nz = jnp.abs(target_f) > 1e-6
    m = fin & nz
    
    diff = jnp.abs(pred_f - target_f) / jnp.clip(jnp.abs(target_f), a_min=1e-6)
    sum_diff = jnp.sum(diff * m)
    total = jnp.sum(m)
    mape_finite = jnp.where(total > 0, sum_diff / jnp.clip(total, a_min=1e-6), jnp.nan)
    
    metrics = {
        "val_ema": ema,
        "val/exact_match_bits": ema,
        "val_nan_acc": nan_acc,
        "val/mape_finite": mape_finite,
    }
    
    metrics = jax.lax.pmean(metrics, axis_name='batch')
    return metrics

def main():
    if config.WANDB_PROJECT and wandb.run is None:
        wandb.init(project=config.WANDB_PROJECT, name=config.WANDB_RUN_NAME)
        
    num_devices = jax.local_device_count()
    print(f"Running on {num_devices} devices.")
    
    rng = jax.random.PRNGKey(config.SEED)
    rng, init_rng = jax.random.split(rng)
    
    state = create_train_state(init_rng)
    state = flax.jax_utils.replicate(state)
    
    step = 0
    for epoch in range(config.EPOCHS):
        with tqdm(total=config.STEPS_PER_EPOCH, desc=f"Epoch {epoch+1}/{config.EPOCHS}") as pbar:
            for _ in range(config.STEPS_PER_EPOCH):
                rng, step_rng = jax.random.split(rng)
                step_rngs = jax.random.split(step_rng, num_devices)
                
                state, metrics = train_step(state, step_rngs)
                
                if step % 50 == 0:
                    log_metrics = {k: float(v[0]) for k, v in metrics.items()}
                    wandb.log(log_metrics, step=step)
                    pbar.set_postfix({"loss": log_metrics["train/loss"]})
                
                step += 1
                pbar.update(1)
                
        val_rngs = jax.random.split(rng, num_devices)
        val_metrics = eval_step(state, val_rngs)
        log_val_metrics = {k: float(v[0]) for k, v in val_metrics.items()}
        wandb.log(log_val_metrics, step=step)
        print(f"Validation EMA: {log_val_metrics['val_ema']:.4f}")

if __name__ == "__main__":
    main()
