"""Correctness tests for the flexible MDD target."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from run_flexible_mdd import build_target, load_data, smooth_basis

jax.config.update("jax_enable_x64", True)


class FlexibleMDDTests(unittest.TestCase):
    def setUp(self):
        self.args = SimpleNamespace(
            bins=28,
            grid_min=-4.0,
            grid_max=14.0,
            kernel_length=1.0,
            flex_scale=1.5,
        )
        self.data = load_data(relative_sigma=0.10, minimum_temperature=350.0)
        self.target = build_target(self.args, self.data)

    def test_complete_timed_schedule_is_propagated_before_masking(self):
        self.assertEqual(len(self.data.temperatures_c), 21)
        self.assertEqual(int(self.data.likelihood_mask.sum()), 18)
        self.assertEqual(self.data.temperatures_c[0], 275)
        self.assertEqual(self.data.temperatures_c[-1], 700)
        self.assertAlmostEqual(
            self.data.observed_fraction[self.data.likelihood_mask].sum(),
            0.5466777461727764,
        )

    def test_smooth_basis_excludes_constant_direction(self):
        grid = np.linspace(-4, 14, 28)
        basis = smooth_basis(grid, 1.0)
        self.assertEqual(basis.shape, (28, 27))
        np.testing.assert_allclose(basis.sum(axis=0), 0, atol=1e-10)
        self.assertAlmostEqual(
            float(np.sqrt(np.mean(np.sum(basis**2, axis=1)))), 1.0
        )

    def test_forward_model_is_absolute_not_window_normalized(self):
        _, _, _, transform, forward_all, _ = self.target
        position = jnp.zeros(self.args.bins)
        weights, _ = transform(position)
        prediction = np.asarray(forward_all(position))
        self.assertAlmostEqual(float(np.asarray(weights).sum()), 1.0)
        self.assertTrue(np.all(prediction >= 0))
        self.assertGreater(prediction[~self.data.likelihood_mask].sum(), 0)
        self.assertLess(prediction[self.data.likelihood_mask].sum(), 1.0)
        self.assertLessEqual(prediction.sum(), 1.0)

    def test_autodiff_matches_finite_difference(self):
        *_, log_density = self.target
        position = jnp.linspace(-0.2, 0.2, self.args.bins)
        gradient = np.asarray(jax.grad(log_density)(position))
        epsilon = 1e-5
        for index in (0, 7, 19, 27):
            direction = np.zeros(self.args.bins)
            direction[index] = epsilon
            finite = (
                float(log_density(position + direction))
                - float(log_density(position - direction))
            ) / (2 * epsilon)
            self.assertAlmostEqual(gradient[index], finite, places=5)


if __name__ == "__main__":
    unittest.main()
