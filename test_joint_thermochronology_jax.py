from __future__ import annotations

import jax
import numpy as np

from joint_thermochronology_jax import build_joint_target, load_joint_data


def test_correct_recoil_excluded_normalization():
    data = load_joint_data()
    np.testing.assert_allclose(data.retained_ar39_total, 163.6722)
    np.testing.assert_allclose(
        data.observed_ar39_fraction[data.likelihood_mask].sum(),
        0.8151598133,
        atol=1e-8,
    )


def test_joint_target_is_finite_and_differentiable():
    _, metadata, transform, forward, log_density = build_joint_target(
        bins=12, mode_count=48, quadrature_count=16
    )
    position = np.zeros(metadata["dimension"])
    value, gradient = jax.value_and_grad(log_density)(position)
    ar39, ratios, discrepancy = forward(position)
    weights, ea, temperature, transformed_discrepancy = transform(position)
    assert np.isfinite(float(value))
    assert np.all(np.isfinite(np.asarray(gradient)))
    assert np.all(np.isfinite(np.asarray(ar39)))
    assert np.all(np.isfinite(np.asarray(ratios)))
    np.testing.assert_allclose(np.asarray(weights).sum(), 1.0)
    np.testing.assert_allclose(float(ea), 117.0)
    np.testing.assert_allclose(float(temperature), 0.0)
    np.testing.assert_allclose(float(discrepancy), float(transformed_discrepancy))
