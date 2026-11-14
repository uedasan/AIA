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

    def test_random_phase_errors_are_reproducible(self):
        first = make_interferograms(self.atten, self.scatt, self.phase,
                                    0, 5, step_err=1, rng=12345)
        second = make_interferograms(self.atten, self.scatt, self.phase,
                                     0, 5, step_err=1, rng=12345)
        np.testing.assert_array_equal(first, second)
        self.assertFalse(np.allclose(first, self.images))


if __name__ == "__main__":
    unittest.main()
