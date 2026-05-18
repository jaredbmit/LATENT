"""Train the motion VAE.

Reads:
  storage/data/vae/features/*.npz
  storage/data/vae/norm_stats.npz

Writes:
  storage/data/latents/<run_name>/model.pt
  storage/data/latents/<run_name>/config.json

Usage:
  uv run python scripts/latents/train_vae.py
  uv run python scripts/latents/train_vae.py --L 8 --residual --run_name mvae_residual
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from motion_latent.vae.dataset import MotionChunkDataset
from motion_latent.vae.model import MotionVAE

FEAT_DIR  = Path("storage/data/vae/features")
STATS_PATH = Path("storage/data/vae/norm_stats.npz")
OUT_ROOT   = Path("storage/data/latents")


def sampling_prob(epoch: int, epochs: int, hold_frac: float, anneal_frac: float) -> float:
    """Scheduled-sampling teacher-forcing probability for a given epoch.

    Flat at 1.0 for the first hold_frac of training, linearly annealed to 0.0
    over the next anneal_frac, then flat at 0.0 for the remainder.
    """
    e = epoch / epochs
    if e <= hold_frac:
        return 1.0
    if e <= hold_frac + anneal_frac:
        return 1.0 - (e - hold_frac) / anneal_frac
    return 0.0


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # --- Data ---
    # H=L: each item is an L-step rollout window. The VAE itself is always a
    # one-step transition model; L is purely the supervision/rollout horizon.
    dataset  = MotionChunkDataset(args.features, args.stats, H=args.L)
    L, D     = dataset.H, dataset.D
    train_dl = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    delta_std = dataset.delta_std.to(device)
    print(f"clips: {dataset.n_clips}  items: {len(dataset)}  L={L}  D={D}")

    # --- Model ---
    model = MotionVAE(D=D, latent_dim=args.latent_dim, hidden=args.hidden,
                      variant=args.variant, residual=args.residual).to(device)
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}  "
          f"variant={args.variant}  residual={args.residual}")

    opt       = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-5)

    # --- Output dir ---
    out_dir = OUT_ROOT / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Train ---
    for epoch in range(1, args.epochs + 1):
        beta = args.beta * min(1.0, epoch / args.beta_warmup) if args.beta_warmup > 0 else args.beta
        p    = sampling_prob(epoch, args.epochs, args.ss_hold, args.ss_anneal)

        model.train()
        train_recon = train_kl = train_iso = 0.0
        for chunk, s_t in train_dl:
            chunk, s_t = chunk.to(device), s_t.to(device)
            loss, metrics = model.rollout_loss(chunk, s_t, p=p, beta=beta, alpha=args.alpha,
                                               iso_mode=args.iso_mode, delta_std=delta_std)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_recon += metrics["recon"]
            train_kl    += metrics["kl"]
            train_iso   += metrics["iso"]
        scheduler.step()

        if epoch % args.log_every == 0:
            n = len(train_dl)
            print(
                f"epoch {epoch:4d}  lr={scheduler.get_last_lr()[0]:.2e}  beta={beta:.3f}  p={p:.3f}  "
                f"recon={train_recon/n:.4f}  kl={train_kl/n:.4f}  iso={train_iso/n:.4f}"
            )

    # --- Save ---
    torch.save(model.state_dict(), out_dir / "model.pt")
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update({"L": L, "D": D, "model_type": "one_step_vae"})
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    print(f"saved → {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features",   type=Path,  default=FEAT_DIR)
    ap.add_argument("--stats",      type=Path,  default=STATS_PATH)
    ap.add_argument("--run_name",   type=str,   default="mvae_base")
    ap.add_argument("--L",          type=int,   default=8,
                    help="Rollout / supervision horizon in frames. The VAE is "
                         "always a one-step transition model.")
    ap.add_argument("--variant",    type=str,   default="conditional",
                    choices=["conditional", "unconditional"])
    ap.add_argument("--residual",   action="store_true",
                    help="Decoder predicts delta from s_t; residual added before loss.")
    ap.add_argument("--latent_dim", type=int,   default=16)
    ap.add_argument("--hidden",     type=int,   default=128)
    ap.add_argument("--epochs",     type=int,   default=1000)
    ap.add_argument("--batch_size", type=int,   default=256)
    ap.add_argument("--lr",         type=float, default=1e-3)
    ap.add_argument("--beta",       type=float, default=1e-3)
    ap.add_argument("--alpha",      type=float, default=0.0,
                    help="Weight on pairwise isometry loss. 0 disables.")
    ap.add_argument("--iso_mode",   type=str,   default="target",
                    choices=["target", "delta"],
                    help="Isometry reference: 'target' (next frame s_{t+1}) or "
                         "'delta' (motion increment s_{t+1}-s_t).")
    ap.add_argument("--beta_warmup", type=int,  default=100,
                    help="Epochs to linearly ramp beta from 0 to --beta. 0 disables warmup.")
    ap.add_argument("--ss_hold",    type=float, default=0.2,
                    help="Fraction of epochs held at p=1 (full teacher forcing).")
    ap.add_argument("--ss_anneal",  type=float, default=0.2,
                    help="Fraction of epochs over which p is linearly annealed 1->0.")
    ap.add_argument("--log_every",  type=int,   default=25)
    train(ap.parse_args())


if __name__ == "__main__":
    main()
