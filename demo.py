"""Reproduce the README figures without a Notebook server or display."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from skimage import data, exposure

from aia import aia, component01, make_interferograms


TAGS = ("Attenuation", "Scattering", "Phase [rad]")


def simulate(seed=12345, steps=5, kx=10, step_err=1.0):
    """Return source, background, and two independently shifted image stacks."""
    rng = np.random.default_rng(seed)
    atten = data.moon() / 256
    low, high = np.percentile(atten, (2, 98))
    scatt = exposure.rescale_intensity(atten, in_range=(low, high)) * 0.8
    phase = exposure.equalize_adapthist(atten, clip_limit=0.03)
    atten += 1
    phase -= phase.mean()

    ny, nx = atten.shape
    py, px = np.mgrid[-ny // 2:ny // 2, -nx // 2:nx // 2]
    envelope = np.exp(-(px**2 + py**2) / 256**2)
    background = (10 * envelope, 0.5 * envelope, 0.2 * envelope)
    atten0, scatt0, phase0 = background
    fg = make_interferograms(
        atten * atten0, scatt * scatt0, phase + phase0,
        kx, steps, step_err, rng,
    )
    bg = make_interferograms(atten0, scatt0, phase0, kx, steps, step_err, rng)
    return (atten, scatt, phase), background, fg, bg


def recover_uniform(fg, bg):
    """Background-normalize Fourier components assuming equal phase steps."""
    fg0, fg1 = component01(fg)
    bg0, bg1 = component01(bg)
    atten = fg0 / bg0
    scatt = np.abs(fg1 / bg1) / atten
    phase = -np.angle(fg1 / bg1)
    phase -= phase.mean()
    return atten, scatt, phase


def recover_aia(fg, bg, *, progress=True):
    """Fit foreground/background independently and remove the background."""
    fga, fgs, fgp = aia(fg, progress=progress)
    bga, bgs, bgp = aia(bg, progress=progress)
    atten = fga / bga
    scatt = fgs / bgs
    phase = np.angle(np.exp(1j * fgp) / np.exp(1j * bgp))
    phase -= phase.mean()
    return atten, scatt, phase


def plot_components(images, path):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3), layout="constrained")
    for ax, image, tag in zip(axes, images, TAGS):
        ax.set_title(tag)
        im = ax.imshow(image, cmap="gray")
        ax.set_axis_off()
        fig.colorbar(im, ax=ax)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_interferograms(fg, bg, path):
    steps = len(fg)
    fig, axes = plt.subplots(2, steps, figsize=(3 * steps, 6), squeeze=False,
                             layout="constrained")
    for row, stack, name in zip(axes, (fg, bg), ("Foreground", "Background")):
        row[0].set_ylabel(name)
        for i, (ax, image) in enumerate(zip(row, stack), start=1):
            ax.set_title(f"Step {i}")
            ax.imshow(image, cmap="gray")
            ax.set_xticks([])
            ax.set_yticks([])
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_comparison(source, uniform, recovered, path):
    fig, axes = plt.subplots(3, 3, figsize=(12, 10), layout="constrained")
    for row, images, name in zip(axes, (source, uniform, recovered),
                                  ("Ground truth", "Uniform steps", "AIA")):
        row[0].set_ylabel(name)
        for ax, image, reference, tag in zip(row, images, source, TAGS):
            ax.set_title(tag)
            im = ax.imshow(image, cmap="gray", vmin=reference.min(),
                           vmax=reference.max())
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def rmse(reference, estimate):
    return {name: float(np.sqrt(np.mean((actual - expected)**2)))
            for name, expected, actual in zip(("attenuation", "scattering", "phase"),
                                               reference, estimate)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--kx", type=float, default=10)
    parser.add_argument("--step-error", type=float, default=1.0,
                        help="standard deviation of phase-step errors in radians")
    parser.add_argument("--save-arrays", action="store_true")
    args = parser.parse_args()
    if args.steps < 3:
        parser.error("--steps must be at least 3")
    if args.step_error < 0:
        parser.error("--step-error must be nonnegative")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source, background, fg, bg = simulate(args.seed, args.steps, args.kx, args.step_error)
    uniform = recover_uniform(fg, bg)
    recovered = recover_aia(fg, bg)
    plot_components(source, args.output_dir / "source.png")
    plot_components(background, args.output_dir / "background.png")
    plot_interferograms(fg, bg, args.output_dir / "interferograms.png")
    plot_comparison(source, uniform, recovered, args.output_dir / "comparison.png")
    metrics = {"parameters": {"seed": args.seed, "steps": args.steps,
                               "kx": args.kx, "step_error": args.step_error},
               "rmse": {"uniform": rmse(source, uniform), "aia": rmse(source, recovered)}}
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    if args.save_arrays:
        np.savez_compressed(args.output_dir / "reconstruction.npz", fg=fg, bg=bg,
                            **{name: value for name, value in zip(
                                ("attenuation", "scattering", "phase"), recovered)})
    print(json.dumps(metrics, indent=2))
    print(f"Saved figures and metrics to {args.output_dir}")


if __name__ == "__main__":
    main()
