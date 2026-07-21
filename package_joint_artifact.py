"""Create a compact tracked record of the expanded joint analysis."""

from __future__ import annotations

import json
from pathlib import Path
import shutil

import numpy as np


RESULTS = Path("results/joint")
RUN = RESULTS / "ensemble_grid_production"
ANALYSIS = RESULTS / "ensemble_analysis"
MARGINAL_AUDIT = RESULTS / "marginal_audit"
CORRECTED_LAB = Path("results/flexible/corrected_lab_control_ta98")
OUTPUT = Path("artifacts/joint_age_spectrum_20260721")


def copy_json(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(source.read_text())
    destination.write_text(json.dumps(data, indent=2) + "\n")


def primary_cell(run_summary: dict) -> dict:
    for scenario in run_summary["expanded_ensemble"]["scenarios"]:
        if (
            int(scenario["duration_my"]) == 100
            and int(scenario["start_age_before_present_my"]) == 100
        ):
            return scenario
    raise RuntimeError("100 My present-ending scenario is absent")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    run_summary = json.loads((RUN / "summary.json").read_text())
    analysis_summary = json.loads((ANALYSIS / "summary.json").read_text())
    copy_json(RUN / "summary.json", OUTPUT / "run_summary.json")
    copy_json(RUN / "run_args.json", OUTPUT / "run_args.json")
    copy_json(
        ANALYSIS / "summary.json", OUTPUT / "analysis_summary.json"
    )
    shutil.copy2("joint_excursion_grid.json", OUTPUT / "scenario_grid.json")
    shutil.copy2("JOINT_AGE_SPECTRUM_MODEL.md", OUTPUT / "MODEL.md")
    for figure in ANALYSIS.glob("*.png"):
        shutil.copy2(figure, OUTPUT / figure.name)
    copy_json(
        MARGINAL_AUDIT / "summary.json",
        OUTPUT / "marginal_audit_summary.json",
    )
    for figure in MARGINAL_AUDIT.glob("*.png"):
        shutil.copy2(figure, OUTPUT / figure.name)
    copy_json(
        CORRECTED_LAB / "manifest.json",
        OUTPUT / "corrected_lab_manifest.json",
    )
    copy_json(
        CORRECTED_LAB / "final_diagnostics.json",
        OUTPUT / "corrected_lab_diagnostics.json",
    )

    posterior = np.load(RUN / "posterior.npz")
    selected_draws = np.linspace(
        0, posterior["temperature_c"].shape[0] - 1, 100, dtype=int
    )
    np.savez_compressed(
        OUTPUT / "compact_posterior.npz",
        **{
            key: posterior[key][selected_draws]
            for key in (
                "position",
                "temperature_c",
                "ea_kj_mol",
                "discrepancy",
                "weights",
                "scenario_responsibility",
                "sampler_stats",
            )
        },
    )
    corrected = np.load(CORRECTED_LAB / "posterior.npz")
    corrected_draws = np.linspace(
        0, corrected["position"].shape[0] - 1, 100, dtype=int
    )
    np.savez_compressed(
        OUTPUT / "corrected_lab_compact_posterior.npz",
        position=corrected["position"][corrected_draws],
        sampler_stats=corrected["sampler_stats"][corrected_draws],
        grid=corrected["grid"],
        basis=corrected["basis"],
        base_logits=corrected["base_logits"],
    )

    primary = primary_cell(run_summary)
    marginal_audit = json.loads(
        (MARGINAL_AUDIT / "summary.json").read_text()
    )
    reweighting = marginal_audit["importance_reweighting_old_to_corrected"]
    scenario_overlap_minimum = min(
        values["overlap_corrected_vs_joint_across_21_scenarios"]["minimum"]
        for values in marginal_audit["shared_marginals"].values()
    )
    interval = primary["temperature_c"]["central_95_and_median"]
    diagnostics = (
        f"max R-hat {analysis_summary['all_parameter_max_rhat']:.4f}; "
        f"minimum bulk/tail ESS "
        f"{analysis_summary['all_parameter_min_bulk_ess']:.0f}/"
        f"{analysis_summary['all_parameter_min_tail_ess']:.0f}; "
        f"minimum cell responsibility ESS "
        f"{analysis_summary['minimum_responsibility_ess']:.0f}; "
        f"divergences {analysis_summary['divergences']}; "
        f"minimum BFMI {analysis_summary['minimum_bfmi']:.3f}"
    )
    readme = f"""# Joint Nakhla age-spectrum analysis

This artifact supersedes the threshold-based temperature transform in
`artifacts/flexible_mdd_20260720`. It fits the laboratory 39Ar release and the
natural normalized radiogenic-Ar spectrum in one likelihood. No fixed argon-
loss criterion is used to infer temperature.

One expanded-ensemble NUTS run marginalizes over the mutually exclusive
duration/onset grid. Responsibility-weighted samples recover the exact
posterior conditional on each grid cell without refitting the same laboratory
experiment or counting the observed natural spectrum more than once.

For a 100 My excursion ending at the present, under a uniform -100 to 100 C
temperature prior, the central 95% interval and median are
`[{interval[0]:.2f}, {interval[1]:.2f}, {interval[2]:.2f}] C`; the conditional
posterior probability above 0 C is
`{primary['temperature_c']['probability_above_zero']:.5f}`.

Sampler diagnostics: {diagnostics}.

The hex-comet animation in the preceding flexible-MDD artifact came from a
different laboratory target: it normalized by all extracted 39Ar and used the
legacy spherical-release approximation. Importance reuse fails decisively
(Pareto k `{reweighting['pareto_k']:.2f}`, ESS
`{reweighting['importance_ess']:.2f}` of {reweighting['samples']:,}). A
separate corrected laboratory-only control passed its sampler gates. Its six
shared diffusion/activation-energy marginals overlap the joint conditionals by
at least `{100 * scenario_overlap_minimum:.1f}%` across all 21 timing cells.
See `marginal_audit_summary.json` and the two marginal-comparison figures.

The cold lower tail is prior-dominated. Event timing changes the upper
constraint and is reported in `joint_temperature_density_grid.png`. This is a Nakhla
analysis covering the last 1.3 Gy; it does not test Noachian denudation or
replace the ALH84001 part of Shuster and Weiss (2005).

`compact_posterior.npz` contains 100 deterministic draws per chain for
inspection. Full checkpoints are intentionally not tracked.
"""
    (OUTPUT / "README.md").write_text(readme)


if __name__ == "__main__":
    main()
