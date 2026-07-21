"""Differentiable joint target for laboratory release and natural Ar spectra."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import pandas as pd

from joint_thermochronology import (
    K40_DECAY_MY,
    SECONDS_PER_MY,
    _value_and_parenthetical_sd,
    load_nakhla1_age_spectrum,
)
EA_MEAN = 117.0
EA_SD = 5.4
LOGD_BASE_MEAN = 5.7
LOGD_BASE_SD = 2.0
R_GAS = 8.314

jax.config.update("jax_enable_x64", True)


def smooth_basis(grid: np.ndarray, kernel_length: float) -> np.ndarray:
    distances = grid[:, None] - grid[None, :]
    kernel = np.exp(-0.5 * (distances / kernel_length) ** 2)
    projector = np.eye(len(grid)) - np.ones((len(grid), len(grid))) / len(grid)
    centered = projector @ kernel @ projector
    eigenvalues, eigenvectors = np.linalg.eigh(centered)
    keep = eigenvalues > max(eigenvalues.max() * 1e-10, 1e-12)
    basis = eigenvectors[:, keep] * np.sqrt(eigenvalues[keep])[None, :]
    marginal_sd = np.sqrt(np.mean(np.sum(basis**2, axis=1)))
    return basis / marginal_sd


@dataclass(frozen=True)
class JointData:
    temperatures_c: np.ndarray
    temperatures_k: np.ndarray
    durations_s: np.ndarray
    observed_ar39_fraction: np.ndarray
    ar39_sigma: np.ndarray
    normalized_ratio: np.ndarray
    normalized_ratio_sigma: np.ndarray
    likelihood_mask: np.ndarray
    retained_ar39_total: float


def load_joint_data(
    relative_ar39_sigma: float = 0.10,
    minimum_temperature_c: float = 350.0,
    normalization_maximum_temperature_c: float = 850.0,
) -> JointData:
    fitted = pd.read_csv("data/nakhla1_parsed_fitted.csv")
    retained = fitted["Temp"] <= normalization_maximum_temperature_c
    retained_total = float(fitted.loc[retained, "39Ar"].sum())
    timed = fitted.dropna(subset=["seconds_per_extraction_step"]).copy()
    observed = timed["39Ar"].to_numpy(float) / retained_total
    measured = timed["std_39Ar"].to_numpy(float) / retained_total
    sigma = np.sqrt(measured**2 + (relative_ar39_sigma * observed) ** 2)
    temperatures_c = timed["Temp"].to_numpy(float)

    age_data = load_nakhla1_age_spectrum(
        minimum_temperature_c=float(temperatures_c.min()),
        maximum_temperature_c=float(temperatures_c.max()),
    )
    if not np.array_equal(age_data.temperatures_c, temperatures_c):
        raise ValueError("age-spectrum and timed laboratory steps do not align")
    likelihood_mask = temperatures_c >= minimum_temperature_c
    return JointData(
        temperatures_c=temperatures_c,
        temperatures_k=temperatures_c + 273.15,
        durations_s=timed["seconds_per_extraction_step"].to_numpy(float),
        observed_ar39_fraction=observed,
        ar39_sigma=sigma,
        normalized_ratio=age_data.normalized_ratio,
        normalized_ratio_sigma=age_data.normalized_ratio_sd,
        likelihood_mask=likelihood_mask,
        retained_ar39_total=retained_total,
    )


def spherical_release(progress: jax.Array, mode_count: int = 64) -> jax.Array:
    progress = jnp.asarray(progress)
    safe_progress = jnp.maximum(progress, 1e-300)
    short = 6.0 * jnp.sqrt(safe_progress / jnp.pi) - 3.0 * progress
    n = jnp.arange(1, mode_count + 1, dtype=progress.dtype)
    weights = 6.0 / (jnp.pi**2 * n**2)
    eigenvalues = jnp.pi**2 * n**2
    long = 1.0 - jnp.sum(
        weights * jnp.exp(-progress[..., None] * eigenvalues), axis=-1
    )
    return jnp.clip(jnp.where(progress < 0.05, short, long), 0.0, 1.0)


def _during_event_production(
    mode_rate_per_my: jax.Array,
    event_start_my: float,
    event_duration_my: float,
) -> jax.Array:
    delta = mode_rate_per_my - K40_DECAY_MY
    x = delta * event_duration_my
    close = jnp.abs(x) < 1e-5
    ratio_series = event_duration_my * (1.0 - x / 2.0 + x**2 / 6.0)
    safe_delta = jnp.where(close, 1.0, delta)
    ratio = jnp.where(
        close,
        ratio_series,
        -jnp.expm1(-x) / safe_delta,
    )
    return (
        K40_DECAY_MY
        * jnp.exp(-K40_DECAY_MY * (event_start_my + event_duration_my))
        * ratio
    )


def build_joint_target(
    bins: int = 28,
    grid_min: float = -4.0,
    grid_max: float = 14.0,
    kernel_length: float = 1.0,
    flex_scale: float = 1.5,
    event_duration_my: float = 100.0,
    event_end_lookback_my: float = 0.0,
    rock_age_my: float = 1300.0,
    temperature_min_c: float = -100.0,
    temperature_max_c: float = 100.0,
    discrepancy_median: float = 0.03,
    discrepancy_log_sd: float = 0.7,
    relative_ar39_sigma: float = 0.10,
    mode_count: int = 96,
    quadrature_count: int = 32,
):
    """Build transforms, predictions, and the normalized joint log density."""
    if event_duration_my <= 0:
        raise ValueError("event duration must be positive")
    event_start_my = rock_age_my - event_end_lookback_my - event_duration_my
    if event_start_my < 0:
        raise ValueError("thermal event begins before closure")
    data = load_joint_data(relative_ar39_sigma)
    grid_np = np.linspace(grid_min, grid_max, bins)
    basis_np = smooth_basis(grid_np, kernel_length)
    base_logits_np = -0.5 * ((grid_np - LOGD_BASE_MEAN) / LOGD_BASE_SD) ** 2

    grid = jnp.asarray(grid_np)
    basis = jnp.asarray(basis_np)
    base_logits = jnp.asarray(base_logits_np)
    temperatures_k = jnp.asarray(data.temperatures_k)
    durations_s = jnp.asarray(data.durations_s)
    observed_ar39 = jnp.asarray(data.observed_ar39_fraction)
    ar39_sigma = jnp.asarray(data.ar39_sigma)
    observed_ratio = jnp.asarray(data.normalized_ratio)
    ratio_sigma = jnp.asarray(data.normalized_ratio_sigma)
    mask = jnp.asarray(data.likelihood_mask)
    mode_n = jnp.arange(1, mode_count + 1, dtype=jnp.float64)
    mode_weights = 6.0 / (jnp.pi**2 * mode_n**2)
    eigenvalues = jnp.pi**2 * mode_n**2
    nodes_np, quadrature_weights_np = np.polynomial.legendre.leggauss(
        quadrature_count
    )
    elapsed = 0.5 * event_duration_my * (jnp.asarray(nodes_np) + 1.0)
    integration_weights = 0.5 * event_duration_my * jnp.asarray(
        quadrature_weights_np
    )

    def transform(position: jax.Array):
        latent = position[: bins - 1]
        ea = EA_MEAN + EA_SD * position[bins - 1]
        logits = base_logits + flex_scale * (basis @ latent)
        weights = jax.nn.softmax(logits)
        temperature_unit = jax.nn.sigmoid(position[bins])
        temperature_c = temperature_min_c + (
            temperature_max_c - temperature_min_c
        ) * temperature_unit
        discrepancy = discrepancy_median * jnp.exp(
            discrepancy_log_sd * position[bins + 1]
        )
        return weights, ea, temperature_c, discrepancy

    def forward(position: jax.Array):
        weights, ea, temperature_c, discrepancy = transform(position)
        lab_rates = jnp.exp(
            grid[:, None] - ea * 1e3 / (R_GAS * temperatures_k[None, :])
        )
        cumulative = jnp.cumsum(lab_rates * durations_s[None, :], axis=1)
        previous = jnp.concatenate(
            [jnp.zeros((bins, 1), dtype=cumulative.dtype), cumulative[:, :-1]],
            axis=1,
        )
        exact_step_release = spherical_release(cumulative) - spherical_release(
            previous
        )
        ar39_prediction = weights @ exact_step_release

        natural_rates = jnp.exp(
            grid - ea * 1e3 / (R_GAS * (temperature_c + 273.15))
        ) * SECONDS_PER_MY
        mode_rates = natural_rates[:, None] * eigenvalues[None, :]
        pre_event = 1.0 - jnp.exp(-K40_DECAY_MY * event_start_my)
        during_without_loss = jnp.exp(
            -K40_DECAY_MY * event_start_my
        ) - jnp.exp(-K40_DECAY_MY * (event_start_my + event_duration_my))
        during_with_loss = _during_event_production(
            mode_rates, event_start_my, event_duration_my
        )
        correction = (
            pre_event
            * jnp.expm1(
                -eigenvalues[None, :]
                * natural_rates[:, None]
                * event_duration_my
            )
            + during_with_loss
            - during_without_loss
        )
        lab_mode_release = (
            jnp.exp(-eigenvalues[None, :, None] * previous[:, None, :])
            - jnp.exp(-eigenvalues[None, :, None] * cumulative[:, None, :])
        )
        correction_release = jnp.einsum(
            "d,dn,n,dnj->j",
            weights,
            correction,
            mode_weights,
            lab_mode_release,
        )
        total_produced = 1.0 - jnp.exp(-K40_DECAY_MY * rock_age_my)
        ar40_release = total_produced * ar39_prediction + correction_release

        pre_retained = pre_event * (
            1.0
            - spherical_release(natural_rates * event_duration_my)
        )
        production = K40_DECAY_MY * jnp.exp(
            -K40_DECAY_MY * (event_start_my + elapsed)
        )
        during_progress = natural_rates[:, None] * (
            event_duration_my - elapsed[None, :]
        )
        during_retained = jnp.sum(
            production[None, :]
            * (1.0 - spherical_release(during_progress))
            * integration_weights[None, :],
            axis=1,
        )
        post_event = jnp.exp(
            -K40_DECAY_MY * (event_start_my + event_duration_my)
        ) - jnp.exp(-K40_DECAY_MY * rock_age_my)
        bulk_ar40 = weights @ (pre_retained + during_retained + post_event)
        ratio_prediction = (ar40_release / ar39_prediction) / bulk_ar40
        return ar39_prediction, ratio_prediction, discrepancy

    def student_t_logpdf(residual: jax.Array, scale: jax.Array, df=4.0):
        standardized = residual / scale
        return (
            jsp.special.gammaln((df + 1.0) / 2.0)
            - jsp.special.gammaln(df / 2.0)
            - 0.5 * jnp.log(df * jnp.pi)
            - jnp.log(scale)
            - 0.5 * (df + 1.0) * jnp.log1p(standardized**2 / df)
        )

    def log_density(position: jax.Array):
        ar39_prediction, ratio_prediction, discrepancy = forward(position)
        ar39_residual = (
            observed_ar39[mask] - ar39_prediction[mask]
        ) / ar39_sigma[mask]
        ar39_likelihood = jnp.sum(
            -0.5 * ar39_residual**2
            - jnp.log(ar39_sigma[mask])
            - 0.5 * jnp.log(2.0 * jnp.pi)
        )
        total_ratio_sigma = jnp.sqrt(ratio_sigma[mask] ** 2 + discrepancy**2)
        ratio_likelihood = jnp.sum(
            student_t_logpdf(
                observed_ratio[mask] - ratio_prediction[mask],
                total_ratio_sigma,
            )
        )
        # Standard-normal priors for the diffusion field, Ea, and log-scale
        # discrepancy. Temperature has a uniform prior on the declared bounds;
        # this is the Jacobian for its logistic unconstraining transform.
        normal_prior = -0.5 * (
            jnp.sum(position[:bins] ** 2) + position[bins + 1] ** 2
        )
        temperature_log_jacobian = (
            jax.nn.log_sigmoid(position[bins])
            + jax.nn.log_sigmoid(-position[bins])
        )
        return (
            normal_prior
            + temperature_log_jacobian
            + ar39_likelihood
            + ratio_likelihood
        )

    metadata = {
        "grid": grid_np,
        "basis": basis_np,
        "base_logits": base_logits_np,
        "dimension": bins + 2,
        "event_start_my": event_start_my,
        "event_duration_my": event_duration_my,
        "event_end_lookback_my": event_end_lookback_my,
        "rock_age_my": rock_age_my,
        "temperature_bounds_c": [temperature_min_c, temperature_max_c],
        "discrepancy_prior": {
            "family": "lognormal",
            "median": discrepancy_median,
            "log_sd": discrepancy_log_sd,
        },
    }
    return data, metadata, transform, forward, log_density
