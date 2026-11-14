"""Phase extraction from phase-shifted interferograms using AIA.

The image stack convention is (steps, height, width). See README.md for the
intensity model, phase convention, and limitations.
"""

import numpy as np
from tqdm import trange


def make_interferograms(atten, scatt, phase, kx, steps, step_err=0, rng=None):
    """Make phase-shifted interferograms with an optional random step error."""
    ny, nx = atten.shape
    px = np.arange(nx)
    carrier = 2 * np.pi * kx * px / nx
    phase_steps = 2 * np.pi * np.arange(1, steps + 1) / steps
    if step_err != 0:
        rng = np.random.default_rng(rng)
        phase_steps = rng.normal(phase_steps, step_err)
    return atten * (1 + scatt * np.cos(phase + carrier + phase_steps[:, None, None]))


def component01(imgs):
    """Return the DC and first harmonic components of equally stepped images."""
    steps, _, _ = imgs.shape
    phase_steps = (2 * np.pi * np.arange(1, steps + 1) / steps)[:, None, None]
    c0 = np.mean(imgs, axis=0)
    c1 = np.mean(2 * imgs * np.exp(1j * phase_steps), axis=0)
    return c0, c1


def aia(imgs, eps=1e-4, n_iter_min=20, n_iter_max=30, *, progress=True):
    """Recover attenuation, scattering, and phase by the AIA iteration."""
    steps, ny, nx = imgs.shape
    imgs = imgs.reshape(steps, -1)

    # Initial phase steps.
    pstep = (2 * np.pi * np.arange(1, steps + 1) / steps)[:, None]

    def solve(n, cosp, sinp, axis):
        A = np.array([
            [n, np.sum(cosp), np.sum(sinp)],
            [np.sum(cosp), np.sum(cosp**2), np.sum(cosp * sinp)],
            [np.sum(sinp), np.sum(sinp * cosp), np.sum(sinp**2)],
        ])
        B = np.array([
            np.sum(imgs, axis=axis),
            np.sum(imgs * cosp, axis=axis),
            np.sum(imgs * sinp, axis=axis),
        ])
        return np.linalg.solve(A, B)

    with trange(n_iter_max, disable=not progress) as bar:
        for i in bar:
            # Stage 0: estimate object terms with fixed phase steps.
            a, b, c = solve(steps, np.cos(pstep), np.sin(pstep), axis=0)
            atten = a
            phase = np.arctan2(-c, b)
            scatt = np.hypot(b, c) / a

            # Stage 1: estimate phase steps with a fixed phase distribution.
            a, b, c = solve(ny * nx, np.cos(phase), np.sin(phase), axis=1)
            prev, pstep = pstep, np.arctan2(-c, b)[:, None]

            prev_rel = prev - prev[0]
            pstep_rel = pstep - pstep[0]
            err = np.sum(np.angle(np.exp(1j * (prev_rel - pstep_rel))) ** 2)
            if err < eps and i > n_iter_min:
                break
            bar.set_postfix(err=f"{err:.4e}")

    return atten.reshape(ny, nx), scatt.reshape(ny, nx), phase.reshape(ny, nx)
