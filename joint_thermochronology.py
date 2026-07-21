"""Forward model for a joint MDD and 40Ar/39Ar age-spectrum analysis.

The functions in this module are deliberately NumPy-first. They provide a
transparent reference implementation against which a JAX sampler can be
tested. Times are in My unless a name explicitly says otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import pandas as pd


SECONDS_PER_MY = 1e6 * 365.25 * 24 * 3600
K40_HALF_LIFE_MY = 1248.0
K40_DECAY_MY = np.log(2.0) / K40_HALF_LIFE_MY
R_GAS = 8.314


@dataclass(frozen=True)
class AgeSpectrum:
    temperatures_c: np.ndarray
    temperatures_k: np.ndarray
    durations_s: np.ndarray
    ar39: np.ndarray
    ar39_sd: np.ndarray
    apparent_age_my: np.ndarray
    apparent_age_sd_my: np.ndarray
    normalized_ratio: np.ndarray
    normalized_ratio_sd: np.ndarray
    bulk_apparent_age_my: float


def _value_and_parenthetical_sd(value: object) -> tuple[float, float]:
    """Parse values such as ``1283 (26)`` or ``7.076 (44)``."""
    text = str(value).strip()
    match = re.fullmatch(r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))(?:\s*\((\d+)\))?", text)
    if not match:
        raise ValueError(f"cannot parse parenthetical uncertainty: {value!r}")
    number_text, uncertainty_text = match.groups()
    number = float(number_text)
    if uncertainty_text is None:
        return number, np.nan
    decimal_places = len(number_text.partition(".")[2])
    uncertainty = int(uncertainty_text) * 10.0 ** (-decimal_places)
    return number, uncertainty


def load_nakhla1_age_spectrum(
    raw_path: Path | str = Path("data/nakhla1.csv"),
    durations_path: Path | str = Path("data/nakhla1_parsed_fitted.csv"),
    minimum_temperature_c: float = 350.0,
    maximum_temperature_c: float = 700.0,
) -> AgeSpectrum:
    """Load corrected apparent ages and reconstructed lab-step durations."""
    raw = pd.read_csv(raw_path)
    durations = pd.read_csv(durations_path)[
        ["Temp", "seconds_per_extraction_step"]
    ]
    frame = raw.merge(durations, on="Temp", how="left", validate="one_to_one")
    valid_age = frame["Age_Ma"].notna()
    all_ages = np.full(len(frame), np.nan)
    all_age_sd = np.full(len(frame), np.nan)
    parsed_all_ages = np.asarray(
        [_value_and_parenthetical_sd(x) for x in frame.loc[valid_age, "Age_Ma"]]
    )
    all_ages[valid_age] = parsed_all_ages[:, 0]
    all_age_sd[valid_age] = parsed_all_ages[:, 1]
    all_ar39 = np.asarray(
        [_value_and_parenthetical_sd(x)[0] for x in frame["39Ar"]]
    )
    ratio = np.expm1(K40_DECAY_MY * all_ages)
    ratio_sd = K40_DECAY_MY * np.exp(K40_DECAY_MY * all_ages) * all_age_sd
    # Shuster & Weiss excluded the four highest-temperature releases affected
    # by implanted/recoiled 39Ar when defining the retained bulk gas.
    bulk_mask = valid_age & (frame["Temp"] <= 850.0)
    bulk_ratio = np.sum(all_ar39[bulk_mask] * ratio[bulk_mask]) / np.sum(
        all_ar39[bulk_mask]
    )
    bulk_age = np.log1p(bulk_ratio) / K40_DECAY_MY
    mask = (
        frame["seconds_per_extraction_step"].notna()
        & frame["Temp"].between(minimum_temperature_c, maximum_temperature_c)
    )
    frame = frame.loc[mask].copy()

    ar39 = np.asarray([_value_and_parenthetical_sd(x) for x in frame["39Ar"]])
    ages = np.asarray([_value_and_parenthetical_sd(x) for x in frame["Age_Ma"]])
    return AgeSpectrum(
        temperatures_c=frame["Temp"].to_numpy(float),
        temperatures_k=frame["Temp"].to_numpy(float) + 273.15,
        durations_s=frame["seconds_per_extraction_step"].to_numpy(float),
        ar39=ar39[:, 0],
        ar39_sd=ar39[:, 1],
        apparent_age_my=ages[:, 0],
        apparent_age_sd_my=ages[:, 1],
        normalized_ratio=ratio[mask] / bulk_ratio,
        normalized_ratio_sd=ratio_sd[mask] / bulk_ratio,
        bulk_apparent_age_my=float(bulk_age),
    )


def spherical_fractional_release(
    progress: np.ndarray, mode_count: int = 128
) -> np.ndarray:
    """Stable spherical diffusion release fraction.

    The short-time expression converges rapidly near zero, while the exact
    eigenmode series converges rapidly away from zero. The legacy sampler used
    the one-mode long-time approximation above progress 0.3; that creates a
    visible numerical discontinuity in accuracy near the switch and is not
    retained in the joint model.
    """
    progress = np.asarray(progress, dtype=float)
    short = 6.0 * np.sqrt(np.maximum(progress, 0.0) / np.pi) - 3.0 * progress
    weights, eigenvalues = spherical_modes(mode_count)
    flat = progress.reshape(-1)
    long = 1.0 - np.sum(
        weights[:, None] * np.exp(-eigenvalues[:, None] * flat[None, :]), axis=0
    )
    long = long.reshape(progress.shape)
    return np.clip(np.where(progress < 0.05, short, long), 0.0, 1.0)


def spherical_modes(count: int) -> tuple[np.ndarray, np.ndarray]:
    n = np.arange(1, count + 1, dtype=float)
    return 6.0 / (np.pi**2 * n**2), np.pi**2 * n**2


def arrhenius_rate_per_second(
    log_d0_over_r2: np.ndarray,
    activation_energy_kj_mol: np.ndarray,
    temperature_k: np.ndarray,
) -> np.ndarray:
    return np.exp(
        np.asarray(log_d0_over_r2)[..., None]
        - np.asarray(activation_energy_kj_mol)[..., None]
        * 1e3
        / (R_GAS * np.asarray(temperature_k))
    )


def laboratory_progress(
    log_d0_over_r2: np.ndarray,
    activation_energy_kj_mol: float,
    temperatures_k: np.ndarray,
    durations_s: np.ndarray,
) -> np.ndarray:
    """Cumulative dimensionless diffusion progress for every domain and step."""
    rates = arrhenius_rate_per_second(
        np.asarray(log_d0_over_r2), activation_energy_kj_mol, temperatures_k
    )
    return np.cumsum(rates * np.asarray(durations_s), axis=-1)


def _during_event_production(
    mode_loss_rate_per_my: np.ndarray,
    event_start_my: float,
    event_duration_my: float,
) -> np.ndarray:
    """Present modal coefficient from argon produced during the excursion."""
    delta = np.asarray(mode_loss_rate_per_my) - K40_DECAY_MY
    x = delta * event_duration_my
    ratio = np.empty_like(delta, dtype=float)
    close = np.abs(x) < 1e-7
    ratio[close] = event_duration_my * (
        1.0 - x[close] / 2.0 + x[close] ** 2 / 6.0
    )
    ratio[~close] = -np.expm1(-x[~close]) / delta[~close]
    return (
        K40_DECAY_MY
        * np.exp(-K40_DECAY_MY * (event_start_my + event_duration_my))
        * ratio
    )


def natural_profile_correction(
    log_d0_over_r2: np.ndarray,
    activation_energy_kj_mol: float,
    event_temperature_c: float,
    event_start_my: float,
    event_duration_my: float,
    mode_count: int = 256,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return deviation from a uniform, closed-system radiogenic-Ar profile.

    The output has shape ``(domain, mode)`` and excludes the spherical mode
    weights. Keeping the uniform closed-system part analytic makes a no-loss
    spectrum exactly flat even at finite modal resolution.
    """
    if event_start_my < 0 or event_duration_my < 0:
        raise ValueError("event timing must be nonnegative")
    b_n, q_n = spherical_modes(mode_count)
    rates_per_my = (
        arrhenius_rate_per_second(
            np.asarray(log_d0_over_r2),
            activation_energy_kj_mol,
            np.asarray([event_temperature_c + 273.15]),
        )[..., 0]
        * SECONDS_PER_MY
    )
    mode_rates = rates_per_my[:, None] * q_n[None, :]
    progress = rates_per_my[:, None] * event_duration_my

    pre_event = 1.0 - np.exp(-K40_DECAY_MY * event_start_my)
    during_without_loss = np.exp(-K40_DECAY_MY * event_start_my) - np.exp(
        -K40_DECAY_MY * (event_start_my + event_duration_my)
    )
    during_with_loss = _during_event_production(
        mode_rates, event_start_my, event_duration_my
    )
    correction = (
        pre_event * np.expm1(-q_n[None, :] * progress)
        + during_with_loss
        - during_without_loss
    )
    return correction, b_n, q_n


def retained_radiogenic_argon(
    log_d0_over_r2: np.ndarray,
    domain_weights: np.ndarray,
    activation_energy_kj_mol: float,
    event_temperature_c: float,
    event_start_my: float,
    event_duration_my: float,
    rock_age_my: float,
    quadrature_count: int = 64,
) -> float:
    """Bulk present-day radiogenic Ar per initial parent abundance."""
    log_d = np.asarray(log_d0_over_r2, dtype=float)
    weights = np.asarray(domain_weights, dtype=float)
    rates_per_my = (
        arrhenius_rate_per_second(
            log_d,
            activation_energy_kj_mol,
            np.asarray([event_temperature_c + 273.15]),
        )[:, 0]
        * SECONDS_PER_MY
    )
    pre = 1.0 - np.exp(-K40_DECAY_MY * event_start_my)
    pre_retained = pre * (
        1.0
        - spherical_fractional_release(rates_per_my * event_duration_my)
    )

    nodes, node_weights = np.polynomial.legendre.leggauss(quadrature_count)
    elapsed = 0.5 * event_duration_my * (nodes + 1.0)
    integration_weights = 0.5 * event_duration_my * node_weights
    production = K40_DECAY_MY * np.exp(
        -K40_DECAY_MY * (event_start_my + elapsed)
    )
    remaining_progress = rates_per_my[:, None] * (
        event_duration_my - elapsed[None, :]
    )
    during_retained = np.sum(
        production[None, :]
        * (1.0 - spherical_fractional_release(remaining_progress))
        * integration_weights[None, :],
        axis=1,
    )
    post = np.exp(
        -K40_DECAY_MY * (event_start_my + event_duration_my)
    ) - np.exp(-K40_DECAY_MY * rock_age_my)
    return float(weights @ (pre_retained + during_retained + post))


def predict_apparent_age_spectrum(
    log_d0_over_r2: np.ndarray,
    domain_weights: np.ndarray,
    activation_energy_kj_mol: float,
    temperatures_k: np.ndarray,
    durations_s: np.ndarray,
    event_temperature_c: float,
    event_start_my: float,
    event_duration_my: float,
    rock_age_my: float = 1300.0,
    mode_count: int = 256,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predict step apparent ages, natural 40Ar release, and 39Ar release."""
    log_d = np.asarray(log_d0_over_r2, dtype=float)
    weights = np.asarray(domain_weights, dtype=float)
    if log_d.ndim != 1 or weights.shape != log_d.shape:
        raise ValueError("log diffusion scales and weights must be matching vectors")
    if not np.isclose(weights.sum(), 1.0) or np.any(weights < 0):
        raise ValueError("domain weights must be nonnegative and sum to one")
    if event_start_my + event_duration_my > rock_age_my:
        raise ValueError("thermal excursion ends after the present")

    cumulative = laboratory_progress(
        log_d, activation_energy_kj_mol, temperatures_k, durations_s
    )
    previous = np.concatenate([np.zeros((len(log_d), 1)), cumulative[:, :-1]], axis=1)
    exact_step_release = spherical_fractional_release(cumulative) - spherical_fractional_release(previous)
    ar39_release = weights @ exact_step_release

    correction, b_n, q_n = natural_profile_correction(
        log_d,
        activation_energy_kj_mol,
        event_temperature_c,
        event_start_my,
        event_duration_my,
        mode_count,
    )
    lab_mode_release = (
        np.exp(-q_n[None, :, None] * previous[:, None, :])
        - np.exp(-q_n[None, :, None] * cumulative[:, None, :])
    )
    correction_release = np.einsum(
        "d,dn,n,dnj->j", weights, correction, b_n, lab_mode_release
    )
    total_produced = 1.0 - np.exp(-K40_DECAY_MY * rock_age_my)
    ar40_release = total_produced * ar39_release + correction_release
    ratio = np.maximum(ar40_release / ar39_release, 0.0)
    apparent_age = np.log1p(np.exp(K40_DECAY_MY * rock_age_my) * ratio) / K40_DECAY_MY
    return apparent_age, ar40_release, ar39_release


def predict_normalized_ratio_spectrum(
    log_d0_over_r2: np.ndarray,
    domain_weights: np.ndarray,
    activation_energy_kj_mol: float,
    temperatures_k: np.ndarray,
    durations_s: np.ndarray,
    event_temperature_c: float,
    event_start_my: float,
    event_duration_my: float,
    rock_age_my: float = 1300.0,
    mode_count: int = 256,
) -> np.ndarray:
    """Predict the Shuster--Weiss observable R/R_bulk for each lab step."""
    _, ar40_release, ar39_release = predict_apparent_age_spectrum(
        log_d0_over_r2,
        domain_weights,
        activation_energy_kj_mol,
        temperatures_k,
        durations_s,
        event_temperature_c,
        event_start_my,
        event_duration_my,
        rock_age_my,
        mode_count,
    )
    bulk = retained_radiogenic_argon(
        log_d0_over_r2,
        domain_weights,
        activation_energy_kj_mol,
        event_temperature_c,
        event_start_my,
        event_duration_my,
        rock_age_my,
    )
    return (ar40_release / ar39_release) / bulk
