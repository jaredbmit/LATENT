"""Train a diffusion model (DiT) directly on normalised motion feature chunks.

Conditioning modes (--cond_mode):
  none         — unconditional generation (default)
  inpaint      — mask the first n_cond frames as known; model denoises [n_cond:]
                 conditioned on noised versions of the context
  prepend      — prepend clean context frames; model denoises [n_cond:] attending
                 to the unnoised prefix
  adaln        — flatten the n_cond context frames to a (n_cond*D,) vector; inject
                 via AdaLN (summed into the timestep embedding before every block)
  input_concat — same flattened vector concatenated channel-wise to every token
                 before in_proj

For inpaint / prepend the model's latent_len is N + H (full [cond|chunk] sequence).
For adaln / input_concat the model's latent_len is H only; conditioning is model-
internal.  The model operates in the VAE-normalised feature space (STATS_PATH);
norm_stats.npz in the run directory is an identity transform (mean=0, std=1) so
rollout scripts that apply a second normalisation step treat it as a no-op.

Reads:
  storage/data/vae/features/*.npz
  storage/data/vae/norm_stats.npz

Writes:
  storage/runs/<run_name>/model.pt
  storage/runs/<run_name>/model_raw.pt
  storage/runs/<run_name>/config.json   (model_type: "motion_dit_raw")

Usage:
  uv run python scripts/diffusion/train_raw_diffusion.py --run_name raw_diff_base --epochs 2000
  uv run python scripts/diffusion/train_raw_diffusion.py --run_name raw_diff_prepend \
      --cond_mode prepend --n_cond 10 --epochs 2000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from motion_latent.paths import FEAT_DIR, RUNS_ROOT, STATS_PATH
from motion_latent.chunk_vae.dataset import MotionChunkDataset
from motion_latent.diffusion.model import MotionDiT, MotionMLP
from motion_latent.diffusion.schedule import cosine_schedule
from motion_latent.features import IDX_JOINT_POS
from motion_latent.kinematics import G1Kinematics


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        self.decay  = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for k, v in model.state_dict().items():
            s = self.shadow[k]
            if v.dtype.is_floating_point:
                s.mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else:
                s.copy_(v)


def train(args: argparse.Namespace) -> None:
    torch.backends.cuda.matmul.fp32_precision   = 'tf32'  # enable TF32 on Ampere+
    torch.backends.cudnn.conv.fp32_precision    = 'tf32'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}  cond_mode={args.cond_mode}  n_cond={args.n_cond}")

    N = args.n_cond   # conditioning frames
    H = args.H        # generative frames

    # --- Load motion chunks and conditioning frames ---
    dataset = MotionChunkDataset(FEAT_DIR, STATS_PATH, H=H, n_cond=max(N, 1))
    D = dataset.D
    print(f"clips: {dataset.n_clips}  items: {len(dataset)}  H={H}  D={D}  N={N}")

    # Build full sequences (N + H, D): [cond | chunk]
    # For unconditional (N=0), seqs = chunks only.
    # MotionChunkDataset already normalises with the VAE norm_stats, so all_seqs
    # is in the normalised feature space the model will operate in.
    all_seqs = []
    for i in range(len(dataset)):
        chunk, cond = dataset[i]       # (H, D), (n_cond, D)
        if N > 0:
            seq = torch.cat([cond[-N:], chunk], dim=0)   # (N+H, D)
        else:
            seq = chunk                                   # (H, D)
        all_seqs.append(seq.numpy())
    all_seqs = np.stack(all_seqs)      # (n_samples, N+H, D)
    n_samples, seq_len, _ = all_seqs.shape
    print(f"sequences: {n_samples}  seq_len={seq_len}")

    # Dataset fits in GPU memory — keep it there and sample minibatches directly
    # to avoid CPU→GPU transfers every step.
    z_gpu      = torch.from_numpy(all_seqs).float().to(device)
    n_samples  = z_gpu.shape[0]
    n_batches  = n_samples // args.batch_size

    # --- Schedule & model ---
    # inpaint / prepend : model sees full [cond|chunk] sequence → latent_len = N + H
    # adaln / input_concat : cond is injected internally → latent_len = H only
    _MODEL_COND_MODES = {"adaln", "input_concat"}
    cond_mode    = args.cond_mode
    backbone     = args.model_type

    if backbone == "mlp" and cond_mode == "adaln":
        raise ValueError("MotionMLP does not support cond_mode='adaln'.")

    model_latent_len = H if cond_mode in _MODEL_COND_MODES else seq_len
    cond_dim         = N * D if cond_mode in _MODEL_COND_MODES else 0

    schedule = cosine_schedule(args.T)
    if backbone == "mlp":
        model = MotionMLP(
            latent_len=model_latent_len, latent_dim=D,
            d_hidden=args.d_model,       n_layers=args.n_layers,
            t_dim=args.t_dim,
            cond_dim=cond_dim,           cond_mode=cond_mode,
        ).to(device)
    else:
        model = MotionDiT(
            latent_len=model_latent_len, latent_dim=D,
            d_model=args.d_model,        n_heads=args.n_heads,
            n_layers=args.n_layers,      ff_mult=args.ff_mult,
            cond_dim=cond_dim,           cond_mode=cond_mode,
        ).to(device)
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}  "
          f"backbone={backbone}  cond_mode={cond_mode}  cond_dim={cond_dim}")
    # model = torch.compile(model)

    opt       = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)
    ema       = EMA(model, args.ema_decay)
    ab        = schedule.alphas_bar.to(device)

    # Geometric (FK) losses operate in unnormalised feature space, so they need
    # the VAE norm stats to map model outputs back to physical units for FK.
    lambda_ee_pos  = args.lambda_ee_pos
    lambda_joint_vel = args.lambda_joint_vel
    fk       = G1Kinematics().to(device) if lambda_ee_pos > 0.0 else None
    mean_t   = torch.tensor(dataset.mean, dtype=torch.float32, device=device)
    std_t    = torch.tensor(dataset.std,  dtype=torch.float32, device=device)
    print(f"lambda_ee_pos={lambda_ee_pos}  lambda_joint_vel={lambda_joint_vel}")

    out_dir = RUNS_ROOT / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    import time
    t0 = time.perf_counter()
    steps_since_log = 0
    aux_sums = {"ee_pos": 0.0, "joint_vel": 0.0}

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        perm = torch.randperm(n_samples, device=device)
        for bi in range(n_batches):
            z0 = z_gpu[perm[bi * args.batch_size : (bi + 1) * args.batch_size]]
            B  = z0.shape[0]
            t   = torch.randint(1, args.T + 1, (B,), device=device)
            ab_t = ab[t].view(B, 1, 1)

            # The model predicts the clean sample x0; loss is MSE(x0_hat, target)
            # on the generative positions only.
            if cond_mode == "none":
                # --- unconditional ---
                eps  = torch.randn_like(z0)
                z_t  = ab_t.sqrt() * z0 + (1 - ab_t).sqrt() * eps
                x0_hat = model(z_t, t)
                loss   = F.mse_loss(x0_hat, z0)
                pred_chunk, tgt_chunk = x0_hat, z0

            elif cond_mode == "inpaint":
                # --- replacement inpainting ---
                eps      = torch.randn_like(z0)                          # (B, N+H, D)
                z_t      = ab_t.sqrt() * z0 + (1 - ab_t).sqrt() * eps
                eps_cond = torch.randn(B, N, D, device=device, dtype=z0.dtype)
                z_t[:, :N] = ab_t.sqrt() * z0[:, :N] + (1 - ab_t).sqrt() * eps_cond
                x0_hat = model(z_t, t)
                loss   = F.mse_loss(x0_hat[:, N:], z0[:, N:])
                pred_chunk, tgt_chunk = x0_hat[:, N:], z0[:, N:]

            elif cond_mode == "prepend":
                # --- clean prefix prepended ---
                cond_frames = z0[:, :N]                                  # (B, N, D) clean
                chunk       = z0[:, N:]                                  # (B, H, D)
                eps         = torch.randn_like(chunk)
                z_t_chunk   = ab_t.sqrt() * chunk + (1 - ab_t).sqrt() * eps
                z_in        = torch.cat([cond_frames, z_t_chunk], dim=1) # (B, N+H, D)
                x0_hat = model(z_in, t)
                loss   = F.mse_loss(x0_hat[:, N:], chunk)
                pred_chunk, tgt_chunk = x0_hat[:, N:], chunk

            elif cond_mode in _MODEL_COND_MODES:
                # --- adaln / input_concat ---
                chunk  = z0[:, N:]                                       # (B, H, D)
                cond_v = z0[:, :N].reshape(B, -1)                       # (B, N*D)
                eps    = torch.randn_like(chunk)
                z_t    = ab_t.sqrt() * chunk + (1 - ab_t).sqrt() * eps
                x0_hat = model(z_t, t, cond=cond_v)
                loss   = F.mse_loss(x0_hat, chunk)
                pred_chunk, tgt_chunk = x0_hat, chunk

            # --- geometric (FK) auxiliary losses ---
            if lambda_ee_pos > 0.0:
                pred = pred_chunk * std_t + mean_t
                tgt  = (tgt_chunk * std_t + mean_t).detach()
                sp = fk(pred[..., IDX_JOINT_POS])
                st = fk(tgt[..., IDX_JOINT_POS])
                ee_pos = (F.mse_loss(sp["hands"], st["hands"])
                        + F.mse_loss(sp["feet"],  st["feet"])) / 2.0
                loss = loss + lambda_ee_pos * ee_pos
                aux_sums["ee_pos"] += ee_pos.item()

            # --- joint-angle velocity loss (normalised space, no FK needed) ---
            if lambda_joint_vel > 0.0:
                dj_pred = pred_chunk[..., IDX_JOINT_POS]
                dj_tgt  = tgt_chunk[..., IDX_JOINT_POS]
                joint_vel_loss = F.mse_loss(dj_pred[:, 1:] - dj_pred[:, :-1],
                                          dj_tgt[:, 1:]  - dj_tgt[:, :-1])
                loss = loss + lambda_joint_vel * joint_vel_loss
                aux_sums["joint_vel"] += joint_vel_loss.item()

            opt.zero_grad(); loss.backward(); opt.step()
            ema.update(model)
            loss_sum += loss.item()
            steps_since_log += 1
        scheduler.step()

        if epoch % args.log_every == 0:
            elapsed = time.perf_counter() - t0
            ms_per_step = 1000 * elapsed / steps_since_log
            msg = (f"epoch {epoch:5d}  lr={scheduler.get_last_lr()[0]:.2e}  "
                   f"loss={loss_sum/n_batches:.5f}")
            steps = n_batches * args.log_every
            active = {k for k, v in aux_sums.items() if v > 0.0}
            if active:
                msg += "  " + "  ".join(f"{k}={aux_sums[k]/steps:.5f}" for k in active)
                for k in active:
                    aux_sums[k] = 0.0
            print(f"{msg}  {ms_per_step:.1f}ms/step")
            t0 = time.perf_counter()
            steps_since_log = 0

    torch.save(ema.shadow,         out_dir / "model.pt")
    torch.save(model.state_dict(), out_dir / "model_raw.pt")
    np.savez(out_dir / "norm_stats.npz",
             mean=np.zeros(D, dtype=np.float32),
             std=np.ones(D, dtype=np.float32))

    cfg = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    cfg.update({
        "model_type": "motion_mlp_raw" if backbone == "mlp" else "motion_dit_raw",
        "latent_len": model_latent_len,
        "H":          H,
        "n_cond":     N,
        "cond_dim":   cond_dim,
        "latent_dim": D,
    })
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    print(f"saved → {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_name",   type=str,   default="v2/rdiff_base")
    ap.add_argument("--H",          type=int,   default=50,
                    help="Chunk length in frames (should match ChunkVAE training).")
    ap.add_argument("--n_cond",     type=int,   default=0,
                    help="Number of conditioning frames prepended (0 = unconditional).")
    ap.add_argument("--cond_mode",  type=str,   default="none",
                    choices=["none", "inpaint", "prepend", "adaln", "input_concat"],
                    help="Conditioning mechanism.")
    ap.add_argument("--model_type", type=str,   default="dit", choices=["dit", "mlp"],
                    help="Denoiser backbone: DiT (transformer) or MLP (no spatial bias).")
    ap.add_argument("--T",          type=int,   default=100)
    ap.add_argument("--d_model",    type=int,   default=128,
                    help="d_model for DiT; d_hidden for MLP. Consider ≥256 for MLP.")
    ap.add_argument("--n_heads",    type=int,   default=4)
    ap.add_argument("--n_layers",   type=int,   default=6)
    ap.add_argument("--t_dim",      type=int,   default=64,
                    help="Timestep embedding dim (MLP only).")
    ap.add_argument("--ff_mult",    type=int,   default=4)
    ap.add_argument("--epochs",     type=int,   default=2000)
    ap.add_argument("--batch_size", type=int,   default=256)
    ap.add_argument("--lr",         type=float, default=1e-4)
    ap.add_argument("--ema_decay",  type=float, default=0.999)
    ap.add_argument("--log_every",  type=int,   default=100)
    # --- auxiliary losses ---
    ap.add_argument("--lambda_ee_pos",  type=float, default=1.0,
                    help="Weight on MSE between predicted/target hand+foot FK positions (pelvis-local).")
    ap.add_argument("--lambda_joint_vel", type=float, default=1.0,
                    help="Weight on MSE of finite-diff joint-angle velocity (29 DOF, normalised space).")
    args = ap.parse_args()

    if args.cond_mode != "none" and args.n_cond == 0:
        ap.error(f"--cond_mode {args.cond_mode} requires --n_cond > 0")
    if args.cond_mode == "none" and args.n_cond > 0:
        ap.error("--n_cond > 0 requires a conditional --cond_mode")

    train(args)


if __name__ == "__main__":
    main()
