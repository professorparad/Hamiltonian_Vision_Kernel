from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Baselines.common import (  # noqa: E402
    DEFAULT_IMAGE_PATH,
    DEFAULT_OUTPUT_ROOT,
    extract_patches,
    load_grayscale_image,
    mse,
    psnr,
    resolve_device,
    save_grayscale,
    save_json,
    seed_everything,
    simple_ssim,
    stitch_patches,
)


class PatchGenerator(nn.Module):
    def __init__(self, latent_channels: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, latent_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(latent_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(patches))


class PatchDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, 1),
        )

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        return self.net(patches)


def train_gan(args: argparse.Namespace) -> dict:
    seed_everything(args.seed)
    device = resolve_device(args.device)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    image_np = load_grayscale_image(args.image_path, args.image_size)
    patches_np = extract_patches(image_np, args.patch_size)
    patch_tensor = torch.from_numpy(patches_np).unsqueeze(1)
    loader = DataLoader(
        TensorDataset(patch_tensor),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )

    generator = PatchGenerator(args.latent_channels).to(device)
    discriminator = PatchDiscriminator().to(device)
    opt_g = torch.optim.Adam(generator.parameters(), lr=args.lr_g, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr_d, betas=(0.5, 0.999))

    history = []
    for epoch in range(args.epochs + 1):
        epoch_g = 0.0
        epoch_d = 0.0
        epoch_rec = 0.0
        batches = 0

        for (real,) in loader:
            real = real.to(device)
            batches += 1

            with torch.no_grad():
                fake_detached = generator(real)
            real_logits = discriminator(real)
            fake_logits = discriminator(fake_detached)
            d_loss_real = F.binary_cross_entropy_with_logits(
                real_logits, torch.ones_like(real_logits)
            )
            d_loss_fake = F.binary_cross_entropy_with_logits(
                fake_logits, torch.zeros_like(fake_logits)
            )
            d_loss = 0.5 * (d_loss_real + d_loss_fake)

            if epoch < args.epochs:
                opt_d.zero_grad()
                d_loss.backward()
                opt_d.step()

            fake = generator(real)
            fake_logits = discriminator(fake)
            rec_loss = F.mse_loss(fake, real)
            adv_loss = F.binary_cross_entropy_with_logits(
                fake_logits, torch.ones_like(fake_logits)
            )
            g_loss = args.reconstruction_weight * rec_loss + args.adversarial_weight * adv_loss

            if epoch < args.epochs:
                opt_g.zero_grad()
                g_loss.backward()
                opt_g.step()

            epoch_g += float(g_loss.detach().cpu())
            epoch_d += float(d_loss.detach().cpu())
            epoch_rec += float(rec_loss.detach().cpu())

        if epoch % args.log_interval == 0 or epoch == args.epochs:
            history.append(
                {
                    "epoch": epoch,
                    "generator_loss": epoch_g / batches,
                    "discriminator_loss": epoch_d / batches,
                    "reconstruction_mse": epoch_rec / batches,
                }
            )

    generator.eval()
    recon_batches = []
    with torch.no_grad():
        for (real,) in DataLoader(TensorDataset(patch_tensor), batch_size=args.batch_size):
            recon_batches.append(generator(real.to(device)).cpu())
    recon_patches = torch.cat(recon_batches, dim=0).squeeze(1).numpy()
    reconstruction_np = stitch_patches(recon_patches, args.image_size, args.patch_size)

    save_grayscale(output_dir / "gan_reconstruction.png", reconstruction_np)
    save_grayscale(output_dir / "gan_target.png", image_np)

    rows_path = output_dir / "gan_training_history.csv"
    with rows_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    metrics = {
        "mse": mse(reconstruction_np, image_np),
        "psnr": psnr(reconstruction_np, image_np),
        "ssim": simple_ssim(reconstruction_np, image_np),
    }
    summary = {
        "algorithm": "GAN patch autoencoder reconstruction baseline",
        "image_path": str(args.image_path),
        "output_dir": str(output_dir),
        "epochs": args.epochs,
        "patch_size": args.patch_size,
        "latent_channels": args.latent_channels,
        "reconstruction_weight": args.reconstruction_weight,
        "adversarial_weight": args.adversarial_weight,
        "metrics": metrics,
    }
    save_json(output_dir / "gan_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a GAN patch-reconstruction baseline."
    )
    parser.add_argument("--image-path", type=Path, default=DEFAULT_IMAGE_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT / "gan")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--latent-channels", type=int, default=32)
    parser.add_argument("--lr-g", type=float, default=0.0008)
    parser.add_argument("--lr-d", type=float, default=0.0004)
    parser.add_argument("--reconstruction-weight", type=float, default=50.0)
    parser.add_argument("--adversarial-weight", type=float, default=1.0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--log-interval", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    summary = train_gan(parse_args())
    print("GAN baseline complete.")
    print(f"Output directory: {summary['output_dir']}")
    print(f"MSE: {summary['metrics']['mse']:.6f}")
    print(f"PSNR: {summary['metrics']['psnr']:.2f}")
    print(f"SSIM: {summary['metrics']['ssim']:.4f}")


if __name__ == "__main__":
    main()
