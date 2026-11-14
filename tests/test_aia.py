"""Analytic checks for the intensity model and phase extraction conventions."""

import unittest

import numpy as np

from aia import aia, component01, make_interferograms


class PhaseExtractionTests(unittest.TestCase):
    def setUp(self):
        y, x = np.mgrid[:32, :48]
        self.atten = 2 + 0.2 * np.sin(2 * np.pi * y / 32)
        self.scatt = 0.3 + 0.1 * np.cos(2 * np.pi * y / 32)
        self.phase = 2 * np.pi * x / 48 + 0.2 * np.sin(2 * np.pi * y / 32)
        self.images = make_interferograms(self.atten, self.scatt, self.phase, 0, 5)

    def test_fourier_components_match_analytic_signal(self):
        dc, harmonic = component01(self.images)
        np.testing.assert_allclose(dc, self.atten, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(harmonic,
                                   self.atten * self.scatt * np.exp(-1j * self.phase),
                                   rtol=1e-12, atol=1e-12)

    def test_aia_recovers_ideal_components_up_to_phase_offset(self):
        atten, scatt, phase = aia(self.images, progress=False)
        np.testing.assert_allclose(atten, self.atten, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(scatt, self.scatt, rtol=1e-12, atol=1e-12)
        difference = np.exp(1j * (phase - self.phase))
        offset = np.angle(difference.mean())
        np.testing.assert_allclose(np.angle(difference * np.exp(-1j * offset)),
                                   0, atol=1e-12)

    def test_aia_recovers_nonuniform_shifts_under_paper_assumptions(self):
        # Spatially constant mean/amplitude satisfy the assumption of paper
        # Step 2. The 3/4-frame shifts come from Tables 2/1, case 1; the
        # phase map here is an analytic fixture, not the paper's simulation.
        y, x = np.mgrid[:12, :32]
        expected_phase = 2 * np.pi * x / 32 + 0.15 * np.sin(2 * np.pi * y / 12)
        shift_sets = (
            [0, 2.1501, 4.2646],
            [0, 4.3497, 1.9003, 2.1561],
            [0, 1.1, 2.7, 4.0, 5.9],
        )
        for shifts in shift_sets:
            with self.subTest(frames=len(shifts)):
                shifts = np.asarray(shifts)
                images = 2 + 0.6 * np.cos(expected_phase + shifts[:, None, None])
                # Force a fixed iteration count; convergence rules are outside
                # this check of the two least-squares updates.
                atten, scatt, phase = aia(
                    images, n_iter_min=100, n_iter_max=100, progress=False,
                )
                np.testing.assert_allclose(atten, 2, atol=1e-10, rtol=0)
                np.testing.assert_allclose(scatt, 0.3, atol=1e-10, rtol=0)
                errors = []
                for sign in (1, -1):
                    difference = np.exp(1j * (phase - sign * expected_phase))
                    offset = np.angle(difference.mean())
                    errors.append(np.max(np.abs(np.angle(
                        difference * np.exp(-1j * offset),
                    ))))
                self.assertLess(min(errors), 1e-10)

    def test_random_phase_errors_are_reproducible(self):
        first = make_interferograms(self.atten, self.scatt, self.phase,
                                    0, 5, step_err=1, rng=12345)
        second = make_interferograms(self.atten, self.scatt, self.phase,
                                     0, 5, step_err=1, rng=12345)
        np.testing.assert_array_equal(first, second)
        self.assertFalse(np.allclose(first, self.images))


if __name__ == "__main__":
    unittest.main()
