#!/usr/bin/env python3
"""Run the documented Lab 4 acceptance example with real SPICE and no agents.

The engine sees only ordinary frozen ExperimentSpec and GeneralCircuitIR objects.
This script supplies example selection, a separate closed-form RC oracle, and plots.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from spiceharness.experiment_sessions import ExperimentManager
from spiceharness.experiments import ExperimentSpec, GeneralCircuitIR

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples" / "experiments"
CANDIDATES = (
    "direct", "original_t", "rc_22ohm_100uf", "lc_10mh_100uf", "damped_lc_10mh_100uf",
    "pi_10mh_100uf", "retuned_t_10mh_100uf", "capacitor_4700uf",
)
LABELS = {
    "direct": "Direct", "original_t": "Literal T", "rc_22ohm_100uf": "RC 22 Ω / 100 µF",
    "lc_10mh_100uf": "LC 10 mH / 100 µF", "damped_lc_10mh_100uf": "Damped LC",
    "pi_10mh_100uf": "π 10 mH / 100 µF", "retuned_t_10mh_100uf": "Retuned T",
    "capacitor_4700uf": "4700 µF / rail",
}


def load_spec(path: Path) -> ExperimentSpec:
    return ExperimentSpec.model_validate_json(path.read_text())


def load_circuit(path: Path) -> GeneralCircuitIR:
    return GeneralCircuitIR.model_validate_json(path.read_text())


def run_comparison(manager: ExperimentManager, spec: Path, circuits: list[Path]) -> dict:
    """Exercise only the public durable-session API; selection stays evaluator owned."""
    record = manager.start_experiment(load_spec(spec), execution_mode="direct")
    run_id = record["run_id"]
    print(f"Started {spec.parent.name}/{spec.name}: {run_id}", flush=True)
    for path in circuits:
        row = manager.evaluate_experiment(run_id, load_circuit(path))
        measured = row["simulation"]
        print(f"  {path.stem}: {measured['status']}; feasible={row['score']['feasible']}; "
              f"packages={row['score']['package_count']}", flush=True)
        if measured["status"] != "ok":
            raise RuntimeError(f"{path}: {measured['errors']}; evidence run {run_id}")
    manager.finalize_experiment(run_id)
    record = manager.get_experiment(run_id)
    if record["status"] != "completed" or not record["result"]["verified"]:
        raise RuntimeError(f"Fresh verification failed: {run_id}")
    return record


def curve(row: dict, analysis: str, source: str | None = None) -> tuple[list[str], np.ndarray]:
    suffix = f"_{analysis}" + (f"_{source}" if source else "") + ".curves.npz"
    directory = Path(row["simulation"]["artifact_dir"])
    paths = list(directory.glob(f"*{suffix}"))
    if len(paths) != 1:
        raise ValueError(f"Expected one {suffix} in {directory}")
    with np.load(paths[0], allow_pickle=False) as saved:
        return saved["names"].tolist(), saved["data"].copy()


def vector(names: list[str], data: np.ndarray, node: str) -> np.ndarray:
    return data[:, names.index(f"v({node})")]


def loaded_rc_oracle(frequency: np.ndarray) -> np.ndarray:
    """Independent complex impedance calculation, including the frozen ESR/load."""
    capacitor = 0.2 + 1 / (2j * np.pi * frequency * 100e-9)
    shunt = 1 / (1 / 10000 + 1 / capacitor)
    return shunt / (1050 + shunt)


def save_figure(figure: plt.Figure, directory: Path, name: str) -> None:
    figure.savefig(directory / f"{name}.svg", bbox_inches="tight")
    figure.savefig(directory / f"{name}.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_evidence(rows: list[dict], directory: Path, oracle_row: dict) -> float:
    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.25})
    figure, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for row in rows:
        name = row["circuit"]["name"]
        names, values = curve(row, "ac", "positive")
        frequency = values[:, 0].real
        axes[0].semilogx(frequency, 20 * np.log10(np.maximum(abs(vector(names, values, "vcc")), 1e-15)),
                         label=LABELS[name])
        axes[1].semilogx(frequency, 20 * np.log10(np.maximum(abs(vector(names, values, "out")), 1e-15)))
    axes[0].axhline(-20, color="black", linestyle=":", label="−20 dB requirement above 1 kHz")
    axes[0].axhline(3, color="gray", linestyle=":")
    axes[0].set(ylabel="Positive rail / source (dB)", title="Measured loaded rail transfer, 10 Hz–100 kHz")
    axes[0].legend(ncol=2, fontsize=8)
    axes[1].set(xlabel="Frequency (Hz)", ylabel="Output / positive source (dB)",
                title="Unvalidated LM741 supply-coupling prediction; not measured device PSRR")
    figure.tight_layout()
    save_figure(figure, directory, "bode")

    selected = [row for row in rows if row["circuit"]["name"] in
                {"direct", "original_t", "rc_22ohm_100uf", "capacitor_4700uf"}]
    figure, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for row in selected:
        names, values = curve(row, "tran")
        time_ms = values[:, 0].real * 1000
        mask = (time_ms >= 10) & (time_ms <= 13)
        if row["circuit"]["name"] == "direct":
            axes[0].plot(time_ms[mask], vector(names, values, "input").real[mask], color="black")
        axes[1].plot(time_ms[mask], vector(names, values, "out").real[mask],
                     label=LABELS[row["circuit"]["name"]])
    axes[0].set(ylabel="Input (V)", title="Nominal oscilloscope-style traces: 2 V DC + 1 V peak at 1 kHz")
    axes[1].set(xlabel="Time (ms)", ylabel="Output (V)",
                title="Actual ngspice transient samples; external random rail noise is not injected here")
    axes[1].legend()
    figure.tight_layout()
    save_figure(figure, directory, "oscilloscope")

    figure, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    x = np.arange(len(rows))
    for ax, metric, title in (
        (axes[0], "prail_noise_rms_v", "Positive rail: integrated external noise contribution"),
        (axes[1], "output_supply_noise_rms_v", "Output: unvalidated macromodel prediction"),
    ):
        ax.bar(x, [row["simulation"]["metrics"][metric] * 1000 for row in rows], color="#246a93")
        ax.set(yscale="log", ylabel="RMS (mV)", title=title)
    axes[1].set_xticks(x, [LABELS[row["circuit"]["name"]] for row in rows], rotation=35, ha="right")
    figure.suptitle("Independent 10 mV RMS white noise per rail, one-sided PSD over 10 Hz–100 kHz")
    figure.tight_layout()
    save_figure(figure, directory, "external-noise")

    names, values = curve(oracle_row, "ac", "signal")
    f = values[:, 0].real
    measured = vector(names, values, "out")
    analytical = loaded_rc_oracle(f)
    error = float(np.max(abs(measured - analytical)))
    if error > 1e-7:
        raise RuntimeError(f"Loaded RC oracle complex transfer error {error} exceeds 1e-7")
    figure, ax = plt.subplots(figsize=(9, 4))
    ax.semilogx(f, 20 * np.log10(abs(measured)), label="ngspice samples")
    ax.semilogx(f, 20 * np.log10(abs(analytical)), "--", label="Independent impedance calculation")
    ax.set(xlabel="Frequency (Hz)", ylabel="Transfer (dB)", title="Reusable loaded RC example")
    ax.legend()
    figure.tight_layout()
    save_figure(figure, directory, "loaded-rc-oracle")
    return error


def write_report(record: dict, bandwidth: dict, oracle: dict, oracle_error: float) -> Path:
    directory = Path(record["result"]["artifacts"]["report"]).parent
    rows = [row for row in record["candidates"] if row["purpose"] == "search"]
    result = record["result"]
    winner = result["circuit"]["name"]
    summary = {
        "execution": "deterministic real ngspice; no agents or provider API calls",
        "run_id": record["run_id"], "bandwidth_run_id": bandwidth["run_id"],
        "loaded_rc_run_id": oracle["run_id"], "winner": winner,
        "fresh_verified": result["verified"], "feasible": result["feasible"],
        "fixture_fingerprint": record["fixture_fingerprint"],
        "model_checksums": result["simulation"]["model_checksums"],
        "loaded_rc_max_complex_error": oracle_error,
        "loaded_t_metrics": bandwidth["result"]["simulation"]["metrics"],
    }
    (directory / "study-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    metrics = list(rows[0]["simulation"]["metrics"])
    with (directory / "comparison.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["candidate", "packages", "feasible", *metrics])
        for row in rows:
            writer.writerow([row["circuit"]["name"], row["score"]["package_count"], row["score"]["feasible"],
                             *(row["simulation"]["metrics"][metric] for metric in metrics)])
    lines = ["# Lab 4: measured acceptance study", "", (f"Run `{record['run_id']}`. Real ngspice; no agents "
             "or provider API calls. The engine's fixture, ranking and measurements remain deterministic."), "",
             (f"Freshly verified winner among eight submitted circuits: **{LABELS[winner]}**, "
             f"{result['score']['package_count']} physical packages; feasible={result['feasible']}. "
             "This is best found under the declared conditions, not a global or practical optimum."), "",
             "## Comparison", "", ("| Candidate | Packages | Pass | Max +rail 1–100 kHz (V/V) | "
             "Peak +rail (V/V) | +drop (mV) | +rail noise (mV RMS) | Output supply noise (µV RMS)* |"),
             "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in rows:
        m = row["simulation"]["metrics"]
        lines.append(f"| {LABELS[row['circuit']['name']]} | {row['score']['package_count']} | "
                     f"{row['score']['feasible']} | {m['pstopband_gain']:.6g} | {m['ppeak_gain']:.6g} | "
                     f"{m['pdrop_v'] * 1000:.5g} | {m['prail_noise_rms_v'] * 1000:.5g} | "
                     f"{m['output_supply_noise_rms_v'] * 1e6:.5g} |")
    lines += ["", ("*Output coupling is an unvalidated manufacturer macromodel prediction. It is not physical "
              "PSRR validation, device internal noise, or an exact Multisim result."), "",
              "![Loaded transfer](bode.png)", "![Nominal transient](oscilloscope.png)",
              "![External noise integration](external-noise.png)", "", "## Literal T bandwidth", "",
              (f"Separate analysis-only run `{bandwidth['run_id']}` gives a falling −3 dB crossing relative "
              f"to 10 Hz of {summary['loaded_t_metrics']['loaded_t_cutoff_hz']:.6g} Hz. Its sampled peak is "
              f"{summary['loaded_t_metrics']['loaded_t_peak_gain']:.6g} V/V. Severe resonance means that "
              "crossing alone does not describe a useful flat passband. Only 10 Hz–100 kHz is swept. "
              "Results are nominal sampled analyses, not a continuous-frequency bound or exhaustive worst case."),
              "", "## Independent reusable check", "",
              (f"Loaded RC run `{oracle['run_id']}` has maximum complex transfer error {oracle_error:.3g} "
              "against an independent impedance calculation over every saved sweep point. "
              "See the repository's `docs/lab4-study.md` for the equation and source interpretation."), "",
              "![Loaded RC oracle](loaded-rc-oracle.png)", "", "## Review evidence", "",
              ("- [Frozen fixture](fixture.json), [fresh selected circuit](circuit.json), "
              "[all saved evidence](evidence.json), [every metric](comparison.csv), [summary](study-summary.json)."),
              ("- Every candidate's `simulation.artifact_dir` contains generated decks, simulator logs, "
              "raw data and CSV/NPZ curves. Model hashes and fixture fingerprints are in the evidence."),
              "- SVG versions of all four figures are adjacent to the PNGs.", "",
              "## Frozen assumptions and limits", ""]
    lines += [f"- {assumption}" for assumption in record["spec"]["assumptions"]]
    lines += ["", ("The large 4700 µF capacitors exploit the assumed 1 Ω source impedance. Their size, cost, "
              "voltage rating, surge current, inrush, leakage and layout have not been constrained. "
              "ESR and DCR are intrinsic part losses, not extra packages. Fixture source impedances/output "
              "load are testbench elements; package counts include the three fixed amplifier packages "
              "and all submitted filter packages."), "", "## Source provenance", ""]
    for source in record["spec"]["provenance"]:
        lines.append(f"- {source['source']}, {source.get('page', 'page unspecified')}: {source['note']}")
    lines += ["", ("Task 3 requires a live NI Multisim demonstration and NI files. This ngspice study does "
              "not satisfy that separately assessed task or produce a Multisim submission."), ""]
    path = directory / "study.md"
    path.write_text("\n".join(lines))
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=ROOT, help="Store local evidence in this workspace")
    args = parser.parse_args()
    manager = ExperimentManager(args.workspace)
    rc = EXAMPLES / "loaded-rc"
    oracle = run_comparison(manager, rc / "spec.json", [rc / "reference.json"])
    lab = EXAMPLES / "lab4-supply"
    bandwidth = run_comparison(manager, lab / "bandwidth-spec.json", [lab / "original_t.json"])
    # Create the main comparison last so the dashboard's default latest run is the answer.
    record = run_comparison(manager, lab / "spec.json", [lab / f"{name}.json" for name in CANDIDATES])
    directory = Path(record["result"]["artifacts"]["report"]).parent
    rows = [row for row in record["candidates"] if row["purpose"] == "search"]
    error = plot_evidence(rows, directory, oracle["candidates"][0])
    report = write_report(record, bandwidth, oracle, error)
    print(f"Run ID: {record['run_id']}\nReport: {report}\n"
          f"Dashboard: uv run spiceharness dashboard {record['run_id']}\n"
          "Execution: real ngspice; no agents or provider API calls", flush=True)


if __name__ == "__main__":
    main()
