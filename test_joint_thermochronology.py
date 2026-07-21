from __future__ import annotations

import numpy as np

from joint_thermochronology import (
    K40_DECAY_MY,
    load_nakhla1_age_spectrum,
    predict_apparent_age_spectrum,
    predict_normalized_ratio_spectrum,
    retained_radiogenic_argon,
    spherical_fractional_release,
    spherical_modes,
)


def representative_setup():
    data = load_nakhla1_age_spectrum()
    return data, np.array([5.7, 9.0]), np.array([0.97, 0.03]), 117.0


def test_parenthetical_data_are_parsed_and_joined():
    data = load_nakhla1_age_spectrum()
    assert len(data.temperatures_c) == 18
    assert data.temperatures_c[[0, -1]].tolist() == [350.0, 700.0]
    assert data.apparent_age_my[[0, -1]].tolist() == [1283.0, 1314.0]
    assert data.apparent_age_sd_my[[0, -1]].tolist() == [26.0, 7.0]
    assert np.all(np.isfinite(data.durations_s))
    np.testing.assert_allclose(data.bulk_apparent_age_my, 1322.4644103)
    np.testing.assert_allclose(
        data.normalized_ratio[[0, 7, -1]],
        [0.9578, 1.0950, 0.9905],
        atol=6e-4,
    )


def test_modal_survival_converges_to_exact_spherical_release():
    progress = np.geomspace(1e-4, 2.0, 80)
    weights, eigenvalues = spherical_modes(4096)
    modal_release = 1.0 - np.sum(
        weights[:, None] * np.exp(-eigenvalues[:, None] * progress[None, :]), axis=0
    )
    exact = spherical_fractional_release(progress)
    assert np.max(np.abs(modal_release - exact)) < 2e-5


def test_no_natural_diffusion_is_exactly_flat_at_closure_age():
    data, log_d, weights, ea = representative_setup()
    ages, _, _ = predict_apparent_age_spectrum(
        log_d,
        weights,
        ea,
        data.temperatures_k,
        data.durations_s,
        event_temperature_c=-273.0,
        event_start_my=900.0,
        event_duration_my=100.0,
        rock_age_my=1300.0,
        mode_count=128,
    )
    np.testing.assert_allclose(ages, 1300.0, rtol=0, atol=1e-9)


def test_hotter_and_longer_events_reduce_early_step_ages():
    data, log_d, weights, ea = representative_setup()
    cold, _, _ = predict_apparent_age_spectrum(
        log_d, weights, ea, data.temperatures_k, data.durations_s,
        -50.0, 900.0, 10.0,
    )
    warm_short, _, _ = predict_apparent_age_spectrum(
        log_d, weights, ea, data.temperatures_k, data.durations_s,
        20.0, 900.0, 10.0,
    )
    warm_long, _, _ = predict_apparent_age_spectrum(
        log_d, weights, ea, data.temperatures_k, data.durations_s,
        20.0, 800.0, 100.0,
    )
    assert warm_short[0] < cold[0]
    assert warm_long[0] < warm_short[0]


def test_closed_system_age_equation_identity():
    age = 1300.0
    produced_per_initial_parent = 1.0 - np.exp(-K40_DECAY_MY * age)
    recovered = np.log1p(
        np.exp(K40_DECAY_MY * age) * produced_per_initial_parent
    ) / K40_DECAY_MY
    np.testing.assert_allclose(recovered, age)


def test_no_loss_normalized_ratio_is_one():
    data, log_d, weights, ea = representative_setup()
    prediction = predict_normalized_ratio_spectrum(
        log_d,
        weights,
        ea,
        data.temperatures_k,
        data.durations_s,
        -273.0,
        900.0,
        100.0,
        1300.0,
    )
    np.testing.assert_allclose(prediction, 1.0, rtol=0, atol=1e-9)


def test_shuster_weiss_hrd_one_percent_boundary():
    total_produced = 1.0 - np.exp(-K40_DECAY_MY * 1300.0)

    def loss(temperature_c):
        retained = retained_radiogenic_argon(
            np.array([5.7]),
            np.array([1.0]),
            117.0,
            temperature_c,
            1200.0,
            100.0,
            1300.0,
        )
        return 1.0 - retained / total_produced

    low, high = -20.0, 10.0
    for _ in range(50):
        middle = 0.5 * (low + high)
        if loss(middle) < 0.01:
            low = middle
        else:
            high = middle
    boundary = 0.5 * (low + high)
    np.testing.assert_allclose(boundary, -7.6080, atol=0.002)
