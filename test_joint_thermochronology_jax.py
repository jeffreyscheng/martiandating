from __future__ import annotations

import jax
import numpy as np

from joint_thermochronology_jax import (
    build_expanded_target,
    build_joint_target,
    load_joint_data,
)


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


def test_one_cell_expanded_target_equals_fixed_target():
    target_options = dict(bins=12, mode_count=48, quadrature_count=16)
    fixed = build_joint_target(
        event_duration_my=100.0,
        event_end_lookback_my=150.0,
        **target_options,
    )
    expanded = build_expanded_target(
        [
            {
                "duration_my": 100.0,
                "start_age_before_present_my": 250.0,
            }
        ],
        **target_options,
    )
    position = np.linspace(-0.2, 0.2, fixed[1]["dimension"])
    fixed_ar39, fixed_ratio, fixed_discrepancy = fixed[3](position)
    expanded_ar39, expanded_ratio, expanded_discrepancy = expanded[3](position)
    np.testing.assert_allclose(expanded_ar39, fixed_ar39, rtol=0, atol=1e-12)
    np.testing.assert_allclose(
        expanded_ratio[0], fixed_ratio, rtol=0, atol=1e-12
    )
    np.testing.assert_allclose(expanded_discrepancy, fixed_discrepancy)
    np.testing.assert_allclose(expanded[5](position), fixed[4](position))
    np.testing.assert_allclose(expanded[4](position), [1.0])


def test_expanded_target_is_finite_and_responsibilities_normalize():
    scenarios = [
        {"duration_my": 10.0, "start_age_before_present_my": 10.0},
        {"duration_my": 100.0, "start_age_before_present_my": 500.0},
        {"duration_my": 500.0, "start_age_before_present_my": 1000.0},
    ]
    _, metadata, _, forward, responsibilities, log_density = (
        build_expanded_target(
            scenarios, bins=12, mode_count=48, quadrature_count=16
        )
    )
    position = np.zeros(metadata["dimension"])
    value, gradient = jax.value_and_grad(log_density)(position)
    _, ratio_predictions, _ = forward(position)
    responsibility = responsibilities(position)
    assert ratio_predictions.shape[0] == len(scenarios)
    assert np.isfinite(float(value))
    assert np.all(np.isfinite(np.asarray(gradient)))
    assert np.all(np.asarray(responsibility) > 0)
    np.testing.assert_allclose(np.asarray(responsibility).sum(), 1.0)
