# Design note: Monte Carlo tolerance analysis (experiments first)

Status: proposal for implementation. Research only; no source was changed. Verified against the code at `12f40a8`
plus the working tree (viewer restyle in progress), `ngspice-47`, and timings measured with
`experiment run` in a scratch workspace.

## 0. Decision summary

| Question | Decision |
| --- | --- |
| Which path | **General experiments** (`ExperimentSpec` / `GeneralCircuitIR`). Template (`DesignSpec`) path deferred to v2 (section 11). |
| What runs | A **post-ranking robustness study** on a saved, valid candidate (default: the verified winner). Option (a) from the brief, with its own budget (b). Never inside ranked evaluation (c). |
| Frozen spec | **Untouched.** No field is added to `ExperimentSpec` or `DesignSpec` (this would change every `fingerprint`). Config lives in a separate `MonteCarloRequest` stored with the study. |
| Perturbed | Candidate-owned plain `R`, `C`, `L` values. Optional multiplicative jitter of `series_resistance_ohm` (off by default). Never fixture loads, sources, `protected_refs`, op-amp/BJT/diode models. |
| Distribution | Default `uniform` in +/- tolerance. Option `gaussian` (sigma = tol/3, truncated at +/- tol). Independent per part. Defaults R 1 %, C 5 %, L 10 %, per-part overrides. |
| RNG | Stateless: each draw is `SHA-256(seed, index, ref, field)` mapped to (0,1). No numpy `Generator`. Sample *i* never depends on N, part order or worker count. |
| Execution | **One ngspice run per sample** through the existing evaluator in a new "lean" mode (no csv/npz/raw kept). No batching, no `.alter`. Optional thread workers. |
| Budget | Own budget, not charged to the 20-evaluation / 180 s session limits. Default N = 100 (max 500), 20 s per chunk, 120 s per study (max 300), 20 s per sample. Chunked and resumable. |
| Output | Per-measurement stats, per-requirement pass fraction + Wilson 95 %, overall yield + Wilson, worst sample, sensitivity, histograms. Saved as a sidecar under the run directory. |
| Ranking | **v1 does not touch ranking, `rank_key`, `finalize_experiment`, or `run.json`.** |

## 1. Evidence from the code (why these decisions)

* `ExperimentSpec.fingerprint = sha256(model_dump_json())` (`experiments.py:312`). A new field, even defaulted,
  changes the dump and therefore every stored `fixture_fingerprint`, the `* fixture <hash>` deck header, and the
  `rescore-kicad` equality check (`experiment_cli.py:384`). Hence a sidecar request, not a spec field.
* `ExperimentManager._evaluate` (`experiment_sessions.py:~170`) appends to `candidates`, bumps `evaluation_count`
  and calls `_active`, which rejects terminal runs. A completed, verified run is exactly the Monte Carlo target, so MC
  must bypass `_evaluate` entirely.
* `ExperimentEvaluator.evaluate` writes, per analysis, `.raw` (ASCII), `.curves.npz`, `.curves.csv` and a copy of the
  model library. Measured: **27 MB per lab4 evaluation** (7 MB of it the transient raw). 200 samples would be about 5 GB.
  The evaluator also spawns `ngspice --version` every call. Lean mode is mandatory.
* Measured time per evaluation (ngspice-47, M-series, includes Python post-processing):
  `loaded-rc` about 0.065 s (op + 1 AC); `lab4-supply` candidates about 0.6 s (op + 3 AC + 1 transient;
  ngspice itself is only about 0.25 s, the rest is raw parsing and writing artifacts). Lean mode should roughly halve
  the lab4 figure (estimate, to be measured by the implementer).
* `generate_netlist` emits literal `{value:.17g}` per component and one analysis per deck; `_read_raw` parses a single
  plot. Batching would need evaluator-emitted control blocks (`.alter`/`foreach`) and a multi-plot parser, and one
  non-converging sample would abort the whole batch. Rejected.
* `_score(circuit, spec, metrics, valid)` already gives `violations`, `feasible` and `rank_key`. MC reuses it so
  per-sample pass/fail is identical to ranking.
* `Component.parameters` must be empty (`domain.py`): the catalog has no safe model-parameter variation. So none.
* Candidate validation already requires protected parts to be bit-equal to the baseline
  (`ExperimentEvaluator.validate`). That is a second line of defence for scope enforcement.

## 2. Scope: what is perturbed

| Part | Perturbed? | Notes |
| --- | --- | --- |
| Candidate `R`, `C`, `L` with a numeric `value`, ref not in `protected_refs` | Yes | `value *= multiplier`. |
| `series_resistance_ohm` of candidate `L`/`C` (DCR/ESR) | Only if `series_resistance_tolerance > 0` (default 0) | Multiplicative, so 0 stays 0. Independent draw, field `"esr"`. Datasheet ESR is usually a typ/max, so off by default. |
| `spec.loads` (RSOURCE, RLOAD ...), `spec.sources`, probes, sweep | **Never** | Frozen bench. They are not in the candidate circuit anyway. |
| `protected_refs` (host op-amp, RF, RG ...) | **Never** | Reject any override that names one. Result states "protected/host parts held at nominal". |
| `OPAMP`, `BJT`, `DIODE` models | **Never** | Vendor model, checksum pinned; no safe parameter hook. Offset, GBW, beta spread are therefore *not* in the yield. |
| `connections` (ideal wires) | Never | |
| Temperature | Never | Frozen at `spec.temperature_c`. |

Override validation (`MonteCarloRequest.validate_against(circuit, spec)`) raises `ValueError` for a `part_tolerances` key
that is unknown, protected, a fixture load, or a non-R/C/L part. Never silently ignore.

## 3. Distributions and defaults

| Field | Default | Range |
| --- | --- | --- |
| `distribution` | `"uniform"` | `"uniform"` or `"gaussian"` |
| `resistor_tolerance` | 0.01 | [0, 0.5] (matches `DesignSpec.resistor_tolerance`) |
| `capacitor_tolerance` | 0.05 | [0, 0.5] (matches `DesignSpec.capacitor_tolerance`) |
| `inductor_tolerance` | 0.10 | [0, 0.5] |
| `series_resistance_tolerance` | 0.0 | [0, 0.5] |
| `part_tolerances` | `{}` | `{ref_lowercase: tol}` overrides the per-kind default for that part |

* `uniform`: `m = 1 + tol * (2u - 1)`. Bounded, conservative (a flat distribution is wider than a binned Gaussian), no
  hidden sigma assumption. This is the default because the user quotes a *guaranteed* tolerance.
* `gaussian`: sigma = tol/3, truncated to +/- tol (3 sigma). Map `u` into `[Phi(-3), Phi(3)]` then
  `z = NormalDist().inv_cdf(u')`, `m = 1 + tol * z / 3`. Truncation keeps every `m` in `[1-tol, 1+tol]`, so values stay > 0.
* `tol == 0` returns exactly `1.0`, and the value is passed through unchanged (not multiplied), so the circuit is bit-equal.
* Perturbed values are rounded with `float(f"{v:.12g}")`.
* All tolerances are **assumptions the client must obtain from the user** (electrolytics are often +/-20 %, asymmetric).
  They are copied into the result (`assumptions`) and into the user-facing summary. Symmetric, independent
  parts only in v1; asymmetric and correlated (matched-pair) tolerances are open questions (section 12).

## 4. Reproducibility (stateless sampler)

New module `src/spiceharness/montecarlo.py`, constant `SAMPLER_VERSION = "mc-sha256-v1"`.

```python
def unit_draw(seed: int, index: int, ref: str, field: str = "value") -> float:
    digest = hashlib.sha256(f"{SAMPLER_VERSION}|{seed}|{index}|{ref.lower()}|{field}".encode()).digest()
    return (int.from_bytes(digest[:8], "big") + 0.5) / 2**64      # strictly inside (0, 1)

def multiplier(u: float, tolerance: float, distribution: str) -> float: ...   # section 3
def sample_multipliers(request, circuit, index) -> dict[str, dict[str, float]]:
    """{ref: {"value": m, "esr": m}} for perturbable parts only, tolerance 0 -> 1.0."""
def perturb_circuit(circuit, multipliers) -> GeneralCircuitIR:
    """Rebuild via GeneralCircuitIR.model_validate(dict) (NOT model_copy: copy skips validation)."""
```

Why not numpy `Generator`: NumPy guarantees the *bit stream* of a BitGenerator, but not the output of `Generator.normal`
across releases. SHA-256 plus stdlib `NormalDist.inv_cdf` is stable across Python/numpy versions and platforms, and gives
random access (worst sample can be rebuilt from `(seed, index)`; chunked, resumed, parallel or enlarged-N studies are
exact prefixes of each other). Same seed + same `ref` yields the same `u` for different candidates (common random numbers),
which makes candidate-to-candidate yield comparisons less noisy.

Persisted per sample: `multipliers` **and** resulting `values`. A fresh rerun redraws and must match exactly
(`repro_check`, section 7). Default `seed = 1` (deterministic CLI/docs/tests). A client wanting an independent replicate
must pass a different seed; the guide says so.

## 5. Where it runs and the budget

### 5.1 Placement

Post-ranking study on one saved candidate (`--top K` in the CLI just loops over feasible leaderboard rows, one study
each). It is not a search evaluation: it does **not** add `candidates`, `evaluation_count`, `elapsed_seconds`, `events`,
`result` or `status` to `run.json`. The ranked search and `finalize_experiment` are unaffected by whether MC ever ran.

Eligibility (`ExperimentManager.run_montecarlo`):

* Run exists, `operation_started_at is None`, `status` not in `{cancelled, cancelling}`.
* `candidate_id=None` means `result.best_candidate_id`, which requires `status == "completed"` and `result.verified`.
  An explicit id (full or unique prefix) must be a saved row with `score.valid` (search or verification).
  Analysis-only runs: `candidate_id` is allowed; default is the finalized selection.
* A fresh **nominal** sample (all multipliers 1.0, same lean path) must be `ok` and match the saved metrics to
  rel 1e-9 or the study stops with `nominal_mismatch` (different ngspice version, edited model, etc.).
* Takes `session.lock` non-blocking for the duration of one chunk (so it cannot overlap `evaluate/finalize`), plus its
  own `study.lock`. If busy: `ValueError("An experiment operation is active; retry when it finishes")`.

### 5.2 Budget numbers

| Parameter | Default | Max | Reason |
| --- | --- | --- | --- |
| `samples` (target N) | 100 | 500 | At N=100 a yield of 95/100 gives Wilson 95 % [0.888, 0.978]. Claiming >= 99 % lower bound needs about 300 samples with no failures (300/300 -> 0.987; 500/500 -> 0.992; 100/100 -> 0.963). |
| `sample_timeout_s` | 20 | 20 | `ExperimentEvaluator` rejects `timeout_s > 20`. Effective value is `min(sample_timeout_s, study_remaining)`. |
| `chunk_seconds` | 20 | 60 | One MCP call does at most this much sampling, then returns progress. Matches the existing 20 s operation model and avoids client tool timeouts. |
| `max_seconds` (study wall clock, cumulative over chunks) | 120 | 300 | N=100 is about 7 s (RC) or about 30-60 s (lab4 class). 500 lab4 samples (about 150-300 s) needs `workers>1`. |
| `workers` | 1 (CLI default `min(4, cpu_count)`) | 8 | See 5.4. |
| fail-fast | first 10 samples all non-`ok` -> `aborted` | | Circuit systematically unmeasurable under jitter; do not burn the cap. |

Execution parameters (`samples`, `max_seconds`, `chunk_seconds`, `workers`, `sample_timeout_s`) are **not** part of the study
identity, so raising `samples` later extends the same study (prefix property). Science parameters are.

### 5.3 Chunking and early stop

`run_montecarlo(run_id, request)` returns after `chunk_seconds`, at `samples` complete, at `max_seconds`, on fail-fast, or
on a `cancel` sentinel file `study.cancel` (checked between samples). The CLI loops over chunks and prints progress;
the MCP client repeats the same call until `state != "running"`. No adaptive CI-based stopping in v1 (it complicates
reproducibility and statistical interpretation). Ctrl-C in the CLI saves the committed prefix and prints the resume command.
`cancel_experiment` is a no-op on terminal runs and is **not** used.

Study `state`: `running` (more chunks due) | `complete` | `truncated` (hit `max_seconds`) | `aborted` (fail-fast) |
`cancelled` | `nominal_mismatch`. Stats are computed over the committed contiguous prefix at every chunk end, so a
`truncated` study is still usable but is labelled with `completed < target`.

Sample status rules: a sample whose own timeout fires with the full 20 s allowance counts as a failed sample
(`status="timeout"`). A sample cut short because the study cap shortened its allowance is **discarded** (not counted,
not committed) to avoid biasing yield. With `workers>1`, indices are processed in blocks of `workers`; a block is committed only
if every sample in it finished, so the committed set is always the contiguous prefix `0..k-1`.

### 5.4 Batching and parallelism

* Batching into one ngspice invocation: **no** (section 1). Each sample is a separate `ngspice -b` process on a deck from the
  existing `generate_netlist`, so no raw SPICE enters anywhere; all decks come from validated `GeneralCircuitIR`.
* Parallelism: safe at process granularity. Each evaluation uses its own temp dir and `HOME=<dir>` (isolated `.spiceinit`), reads
  the model library read-only, and ngspice is single threaded. Use `ThreadPoolExecutor(workers)` (subprocess waits release
  the GIL) with **one `ExperimentEvaluator` per worker thread**. Results are keyed by sample index, so output is identical for
  any `workers` (tested). Memory per sample is small in lean mode.

## 6. Required evaluator change (minimal, default-off)

`ExperimentEvaluator.__init__(..., *, lean: bool = False, simulator_version: str | None = None)`.

When `lean=True` only: skip `ngspice --version` (use the cached/passed version); skip `np.savez_compressed`, `np.savetxt`
and `metrics.json`; copy the model lib once into a per-sample temp dir created under the study `scratch/` and `shutil.rmtree`
it in a `finally`; return `artifact_dir=None, artifacts=()`. Everything else (`validate`, `generate_netlist`, `_read_raw`,
`_measure`, `_envelope`, `_score`, timeouts, NaN checks) runs the **same code path**. With `lean=False` behaviour and
artifact lists must be byte-for-byte as today. Do not duplicate the evaluator.

## 7. Result schema and storage

Location (inside the existing run directory; `run.json` never modified):

```
.spiceharness/experiments/RUN_ID/montecarlo/STUDY_ID/
    study.json      atomic tmp+replace; request, identity, progress, summary (below)
    samples.jsonl   append-only, contiguous, one JSON object per committed sample
    worst/          full-fidelity (non-lean) rerun of the worst sample + circuit.json (created at completion)
    scratch/        temp decks, empty after every sample
    study.lock  study.cancel (sentinel)
```

`STUDY_ID = sha256(canonical JSON of {run_id, candidate_id, fixture_fingerprint, circuit_fingerprint, SAMPLER_VERSION,
seed, distribution, tolerances, part_tolerances, series_resistance_tolerance}).hexdigest()[:16]`. Calling again with the same
science parameters resumes; a new seed/tolerance makes a new study. Use `fcntl.flock` and `encoded()` from `storage.py`,
as `ExperimentManager._save` does. Must write under a real directory (reject symlinks like `_load_saved_run`).

`study.json` (`"schema": "spiceharness.montecarlo/1"`; models are `StrictModel` in `montecarlo.py`):

```
{
 "schema", "study_id", "run_id", "candidate_id", "candidate_name", "created_at", "updated_at",
 "identity": {"fixture_fingerprint", "circuit_fingerprint", "sampler_version", "simulator_version", "model_checksums"},
 "request": {...MonteCarloRequest...},                  # science + execution params as used
 "state": "running|complete|truncated|aborted|cancelled|nominal_mismatch",
 "progress": {"target": 100, "completed": 100, "elapsed_seconds": 7.4, "chunks": 1},
 "analysis_only": false,                                 # copied from the run
 "scoring": "Not scored (analysis only, no acceptance requirements)" | "scored under the frozen requirements",
 "notice": "Statistical estimate over sampled R/L/C tolerances of candidate-owned parts under the frozen fixture; not a worst-case proof. Excludes host/protected-part tolerance, device-model variation, temperature and source tolerance.",
 "assumptions": ["resistor 1% uniform", ...],            # human-readable, from the request
 "nominal": {"metrics": {...}, "feasible": true, "matches_saved": true},
 "counts": {"attempted": 100, "ok": 100, "invalid": 0, "timeout": 0, "failed": 0, "invalid_reasons": {"<error text>": n}},
 "yield": {"passing": 97, "attempted": 100, "fraction": 0.97, "wilson95": [0.915, 0.990],
           "limiting_requirement": "cutoff_hz"} | null,   # null when analysis_only
 "requirements": [{"metric", "index", "minimum", "maximum", "passing", "attempted", "fraction", "wilson95",
                   "min_margin", "min_margin_sample"}],   # [] when analysis_only
 "measurements": {"<name>": {"n", "nominal", "mean", "std", "min", "max",
                             "percentiles": {"1","5","25","50","75","95","99"},
                             "histogram": {"edges": [21 floats], "counts": [20 ints]},
                             "min_sample", "max_sample"}},
 "sensitivity": {"<metric>": [{"ref": "c1", "pearson": -0.91}, ...top 3]},     # n_ok >= 20 only
 "worst_sample": {"index", "max_violation", "violated": ["cutoff_hz"], "multipliers", "values", "metrics",
                  "circuit": {...GeneralCircuitIR...}, "artifact_dir", "reverified": true} | null,
 "invalid_samples": [first 5 {"index","status","error"}],
 "repro_check": {"samples_redrawn": 100, "multipliers_equal": true}
}
```

`samples.jsonl` line: `{"i", "status", "valid", "feasible", "multipliers", "values", "metrics", "violations", "error", "seconds"}`.

Definitions (the implementer must follow these exactly):

* **Sample passes** iff `status=="ok"` and `score.feasible is True` (from `_score`). Invalid/timeout/failed samples are failures
  *and* reported in `counts` (a cutoff pushed outside the declared band raises `ValueError` in `_measure`; that is an invalid
  sample with its reason, not a crash).
* **Yield** = passing / attempted, Wilson score interval, z = 1.959963984540054, clamped to [0, 1]. `None` when attempted == 0.
* **Per-requirement** `passing`: ok sample and `violations[f"{metric}:{index}"] == 0`; non-ok samples count as failing every requirement.
  `min_margin` = min over ok samples of `min(value-minimum, maximum-value)/scale` using the same `scale` as `_score`
  (negative means violated). `limiting_requirement` = fewest passing (ties: lowest index).
* **Measurement stats** use ok samples only; std is `ddof=1`; percentiles use numpy linear interpolation; histogram has 20
  bins over `[min,max]` (a single bin when `min==max`).
* **Worst sample** = ok sample with the largest `max(violations.values())` (ties: lowest index); `null` if no sample violates
  anything (`yield.fraction == 1`) or the run is analysis-only. At completion it is rerun once with the non-lean
  evaluator (artifacts kept under `worst/`) and `reverified` records that its metrics match the lean result to rel 1e-9.
  For analysis-only studies, `min_sample`/`max_sample` per measurement give the extremes instead.
* **Sensitivity** = Pearson correlation between each part's multiplier and each metric over ok samples, top 3 by |r|; labelled
  correlation (sampling-based screening), not a derivative or worst case.
* **repro_check**: at completion redraw all multipliers from `(seed, index)` and assert exact equality with `samples.jsonl`.

## 8. Ranking interaction

v1: **none**. `rank_key`, `leaderboard`, `finalize_experiment`, and winner selection ignore studies. `get_experiment` adds a
`montecarlo` summary list (`study_id`, `candidate_id`, `state`, `completed/target`, `yield`) **only when the directory exists**,
so existing responses are unchanged (response models already use `extra="allow"`).

If ever used (v2): a separately labelled *derived* "robust ordering" built from sidecars: feasible candidates only, by Wilson lower bound
of yield descending, then the saved `rank_key`. It would never replace the saved `rank_key`/winner or rewrite `run.json`. Valid > feasible >
packages > margin > power stays authoritative.

## 9. Surfaces

### 9.1 Code layout

| File | Change |
| --- | --- |
| `src/spiceharness/montecarlo.py` (new) | `MonteCarloRequest`, `unit_draw`, `multiplier`, `sample_multipliers`, `perturb_circuit`, `wilson_interval(k, n)`, `summarize(samples, spec, nominal) -> dict`, `run_chunk(...)`, `MonteCarloStore` (paths, locks, jsonl, atomic study.json), constants `NOTICE`, `SAMPLER_VERSION`. No Typer/MCP imports. |
| `src/spiceharness/experiments.py` | `lean` / `simulator_version` evaluator options (section 6) only. No spec/IR field changes. |
| `src/spiceharness/experiment_sessions.py` | `ExperimentManager.run_montecarlo(run_id, request) -> dict`, `get_montecarlo(run_id, study_id=None) -> dict | list`, `list_montecarlo(run_id)`; `get_experiment` appends `montecarlo` summary only if the dir exists. |
| `src/spiceharness/experiment_cli.py` | `montecarlo` and `montecarlo-show` commands. |
| `src/spiceharness/mcp_server.py` | Tools `run_monte_carlo`, `get_monte_carlo`; `EXPERIMENT_GUIDE` step 8; server `instructions` sentence. |
| `src/spiceharness/results_viewer.py` | Load sidecars into `_viewer_data` (read-only, size-capped, no symlinks); a "Monte Carlo" panel. |
| `docs/VISION.md`, `README.md` | Section 9.5. |
| `tests/test_montecarlo.py` (new) + `tests/test_interfaces.py` | Section 10. |

### 9.2 CLI (`experiment_cli.py`)

```
spice experiment montecarlo RUN_ID [-w WS] [--candidate ID_OR_PREFIX]... [--top K]
      [-n/--samples 100] [--seed 1] [--distribution uniform|gaussian]
      [--r-tol 0.01] [--c-tol 0.05] [--l-tol 0.10] [--esr-tol 0.0] [--tol REF=FRACTION]...
      [--max-seconds 120] [--workers N] [--min-yield F] [--json]
spice experiment montecarlo-show RUN_ID [STUDY_ID_OR_PREFIX] [-w WS] [--json]
```

* Resolve run prefix with `cli._expand_run_id`, candidate prefix like the `candidate` command (unique prefix, error on ambiguity).
  `--candidate` is repeatable, `--top K` (<= 5, scored runs only) selects the first K *feasible* leaderboard rows; omitted = verified winner.
* Prints the notice, assumptions, N/seed, yield + Wilson (or `Not scored (analysis only, no acceptance requirements)`), the
  measurement and requirement tables, worst-sample values, and (for several candidates) one advisory comparison table
  headed "Robustness (statistical, advisory; not the ranking)".
* Exit 0 on a completed study regardless of yield (statistics are not a verdict). `--min-yield F` exits 1 when the Wilson lower
  bound is below F or yield is null. Errors use `_error()`. Direct deterministic driver; text must say "no agents, real ngspice".
* Follow-up hints via `command_hint(f"experiment montecarlo-show {shlex.quote(run_id)} {study_id}", workspace)` and
  `command_hint(f"view {run_id}", workspace)`, as `candidates` does; never hand-build `-w`.

### 9.3 MCP (`mcp_server.py`)

* `run_monte_carlo(run_id: str, request: MonteCarloRequest = MonteCarloRequest()) -> MonteCarloResponse`, annotation `LOCAL_WRITE`.
  Response (`extra="allow"`): `study_id`, `state`, `completed`, `target`, `yield`, `requirements`, `worst_sample`, `notice`,
  `assumptions`, `viewer_command`, `montecarlo_command`, and `next_step` ("call run_monte_carlo again with the same request to
  continue" while `running`; otherwise a reminder to quote the notice and assumptions).
* `get_monte_carlo(run_id: str, study_id: str | None = None)`, annotation `READ_ONLY`; full `study.json` (histograms etc.).
* Update the exact tool set in `tests/test_interfaces.py:121` (add `run_monte_carlo`, `get_monte_carlo`). Check
  `tests/test_experiment_mcp.py:69`, `tests/test_client_mcp.py:36`, `tests/test_kicad_import.py:686` (they build dicts; fix any exact-set assertions).
* `EXPERIMENT_GUIDE` step 8 (text to add): optional robustness study after `finalize_experiment`. **Ask the user for part
  tolerances**; do not assume them. Call `run_monte_carlo` repeatedly with the same request until `state != "running"`. Report
  N, seed, distribution, tolerances, yield with its Wilson interval, the limiting requirement and worst-sample values, and
  state: "statistical estimate, not a worst-case proof; host/protected parts, device models, temperature and sources were held
  at nominal." Analysis-only runs: statistics only, "Not scored (analysis only, no acceptance requirements)". Studies never
  change ranking, the winner or verification. The template `GUIDE` is unchanged in v1.

### 9.4 Viewer (`results_viewer.py`; plain, no colours, another agent is restyling)

`_viewer_data` adds `candidates[i]["montecarlo"] = [study summaries] | None` (key always present, `None` when absent), read
from `experiments/RUN_ID/montecarlo/*/study.json` after the same symlink and "inside `.spiceharness`" checks used by
`_artifacts`; skip files > 2 MiB. Per study the panel shows, as plain tables/text:

1. Heading "Monte Carlo (statistical, not worst-case proof)", notice, assumptions, N / seed / distribution, state, `completed/target`.
2. Yield + Wilson interval, or the analysis-only string.
3. Requirements table: metric, bound, passing/attempted, fraction, Wilson, min margin.
4. Measurements table: nominal, mean, std, min, p5, p50, p95, max.
5. Per bounded metric a text histogram from `histogram.counts` (bars as `#` repeated; no SVG colours required).
6. Worst sample: index, violated metrics, part values vs nominal.

The viewer recomputes nothing; it displays saved numbers only (its module docstring promises this).

### 9.5 Docs

* `docs/VISION.md` Limits: keep the corner sentence, add "Monte Carlo (experiments only) is an opt-in statistical estimate of yield over sampled
  R/L/C tolerances of candidate-owned parts. It is not worst-case proof and excludes host/protected parts, device-model variation and
  temperature." Roadmap: replace "Stronger robustness ranking: Monte Carlo yield, and temperature sweeps as optional ranking criteria" by two lines:
  Monte Carlo yield analysis shipped for experiments (template path next); robustness *ranking* and temperature sweeps remain future work.
  Status section: add one line, update test count.
* `README.md`: a short "Monte Carlo tolerance study" subsection next to the tolerance-corner paragraph (line ~67) with the CLI example.
* `.claude/CLAUDE.md` should get a short paragraph describing `run_monte_carlo`/`experiment montecarlo`; I was told not to edit it, so the
  owner should add: tools, "statistical not worst-case", "ask user for tolerances", "does not affect ranking".

## 10. Tests (`tests/test_montecarlo.py`; `rc_fixture` from `tests/test_experiments.py` can be copied, but use the light AC-only spec for ngspice tests)

Unit (no ngspice; inject a fake evaluator via `run_chunk(..., evaluator_factory=...)` returning closed-form RC metrics):

1. `test_same_seed_identical`: two studies, same request, identical `samples.jsonl` and summary (excluding timestamps).
2. `test_different_seed_differs`: multipliers differ; yields/means differ.
3. `test_sample_independent_of_n_order_workers`: `sample_multipliers(i)` same for N=10 vs 500, shuffled component order, `workers=1` vs `3`; and a study at
   N=20 is an exact prefix of N=40 (extending an existing study continues, no redraw).
4. `test_zero_tolerance_unchanged`: tol 0 -> multiplier exactly `1.0` and `perturb_circuit` value is bit-equal; `part_tolerances={"r1": 0}` leaves R1 untouched while C1 varies.
5. `test_bounds_and_distribution`: uniform within `[1-tol,1+tol]`; gaussian truncated within the same, sample std approximately tol/3 over 20 000 pure draws (rel 5 %).
6. `test_fixture_and_protected_never_perturbed`: use `examples/experiments/lab4-supply/spec.json` (protected `U1, RF, RG`) with a candidate that includes them; every sample keeps those components
   equal to the baseline, loads absent, `ExperimentEvaluator.validate` accepts each sample; overrides naming `U1`, `RLOAD` or an unknown ref raise `ValueError`.
   Defence in depth: a hand-built circuit with `RF` changed makes `validate` raise.
7. `test_wilson_known_values` (tolerance 1e-5): (95,100) -> (0.88825, 0.97846); (10,10) -> (0.72247, 1.0); (0,10) -> (0.0, 0.27753);
   (50,100) -> (0.40383, 0.59617); (0,1) -> (0.0, 0.79345); n=0 -> `None`; (300,300) lower about 0.9874.
8. `test_summary_stats`: hand-made sample list -> mean/std(ddof=1)/percentiles/histogram counts sum to n; per-requirement `passing` and `min_margin`; invalid samples count as failures and are listed in `counts`.
9. `test_analysis_only`: spec with no requirements -> `yield is None`, `requirements == []`, `worst_sample is None`, measurement stats present, `scoring` is exactly "Not scored (analysis only, no acceptance requirements)"; `rank_key`/`feasible` per sample are `None`.
10. `test_budget_and_timeouts`: fake evaluator with sleeps: `max_seconds` small -> `state=="truncated"`, committed prefix contiguous, stats over the prefix; sample timeout -> `status=="timeout"` counted as failed; a sample cut off by the study cap is discarded, not counted;
    first 10 invalid -> `aborted`; chunking: three `chunk_seconds`-limited calls produce the same file as one call; cancel sentinel -> `cancelled`.
11. `test_default_off_changes_nothing`: golden fingerprints (section 11) unchanged; `ExperimentSpec.model_fields` and `GeneralCircuitIR` unchanged (compare to a frozen list); a normal experiment run produces `run.json` with no `montecarlo` key and no `montecarlo/` directory; `get_experiment`
    output has no `montecarlo` key; a run directory saved before the change (copy a recorded `run.json` into `tmp_path`) loads and `finalize` still works.
12. `test_mc_does_not_touch_run`: after a study, `run.json` bytes are identical, `evaluation_count`, `candidates` and `result` unchanged, a second `evaluate_experiment` on a non-terminal run still works.
13. Evaluator: `lean=False` artifact list equals the pre-change listing (same file names); `lean=True` leaves no files in `scratch/` and `artifact_dir is None`.
14. Interfaces: update exact tool set; `run_monte_carlo` returns `state`, `notice`; `get_monte_carlo` is `READ_ONLY`; MCP chunk loop reaches `complete`.
15. CLI (`typer.testing.CliRunner`, tiny N): `experiment montecarlo` prints the notice, supports `--json` schema keys, `--min-yield 1.1` exits 1, ambiguous candidate prefix errors, `montecarlo-show` works; hints contain the workspace quoting.
16. Viewer: with a saved study the payload has `montecarlo` and the HTML contains "statistical, not worst-case proof"; without one the key is `None` and the page is unchanged.

Real ngspice (marked `real_spice`, skip if `ngspice` missing; keep under about 3 s): `loaded-rc` spec + `reference.json` with N=20, seed 1, 4 workers:
every sample's metrics equal an independent non-lean `ExperimentEvaluator.evaluate` of that sample's saved perturbed circuit (rel 1e-9), and the spread of `cutoff_hz` is nonzero and bounded by the +/-(R,C) corner values computed with the same evaluator; pass counts equal a recomputation from `samples.jsonl`;
a second run into a fresh workspace reproduces identical multipliers and metrics (rel 1e-12) and identical yield; `scratch/` empty; study dir < 1 MB.

## 11. Golden values for the default-off test (computed with `uv run python` on this tree)

| File | `fingerprint` |
| --- | --- |
| `examples/experiments/loaded-rc/spec.json` (ExperimentSpec) | `8761039efa075df0def82f15306a4bab8c1d8e79b6868d15655b6c939976c585` |
| `examples/experiments/lab4-supply/spec.json` | `be657fbbe7f551a7b386cfe3d1c6c72788c9a6cf5958a80061cac73c22a93fb5` |
| `examples/experiments/lab4-supply/bandwidth-spec.json` | `f47ed0ca286b0f3f92ebba86713065b1523b5ac720cb1c06de6bf967c9d7a86b` |
| `examples/experiments/interference-filter/spec.json` | `48b2128b580cb63607e45b7904b7591aeab229d66cec4d03053e8b5f4d74e141` |
| `examples/experiments/loaded-rc/reference.json` (GeneralCircuitIR) | `68bffb1e2354dc5354a765a432e2542cff13a30fb93394608183dbd188c2e30e` |
| `examples/experiments/lab4-supply/rc_22ohm_100uf.json` (GeneralCircuitIR) | `27216d91db63ffeeb6f51b96b1859063041518155df3ed0441acc0fc936eb3e8` |
| `examples/amplifier.json` (DesignSpec) | `937e03d88cce5b6de84862696385f3fd1f3dbd987cfe4c89e5b5bd6b70ec4b41` |
| `examples/lowpass.json` (DesignSpec) | `784a1d518c0076a3d36e235f3947b6a49eca04720d9586a4ebf9a4ebd9a25349` |

Re-run the one-liner before implementing in case the working tree moved: load each file with `model_validate_json`, print `.fingerprint`.
Template path (v2) sketch: `DesignSpec` already carries `resistor_tolerance`/`capacitor_tolerance`; reuse them as sampler defaults, reuse
`sample_multipliers`/`summarize`, evaluate with `NgSpiceSimulator.evaluate(circuit.model_copy(update={"components": parts, "spec": nominal_spec}))` as the
corner code does (`simulator.py:~400`), and store the study as a sidecar under `.spiceharness/runs/RUN_ID/montecarlo/`. The scoring there is
`scoring.py`, not `_score`, and `corner_policy` must stay independent. Out of scope for v1.

## 12. Risks and open questions

* **Garbage in, garbage out:** the yield is only as good as the tolerance assumptions. Electrolytic caps (+/-20 % or -20/+80 %), bias, or
  matched/ratio tolerance are not modelled; independent symmetric draws can over- or understate yield. Mitigation: assumptions are mandatory output; the guide
  tells the client to ask. Open: add asymmetric `(low, high)` tolerances and correlated groups in v2?
* **Host parts held nominal:** protected `RF/RG` (lab4) or source impedance never vary, so a gain bound that depends on them looks artificially
  robust. Required wording in the notice. Open question: an explicit, user-approved `host_tolerances` request field that perturbs protected parts
  *in the study only* (never in a candidate). Not recommended for v1 since it contradicts "never perturb protected parts".
* **Device-model variation absent:** offset, GBW, beta spread, supply drift. A feasible-by-MC circuit can still fail on real silicon.
* **Time on heavy circuits:** transient-dominated decks (0.3-0.6 s/sample). Mitigations: lean mode, workers, resumable chunks, `max_seconds`. The
  existing "7+ passives corners > 20 s" template issue is not made worse because MC samples are separate processes with their own budget. Needs measurement of the real lean speed-up.
* **Lean path divergence:** the same code path with I/O switched off; the worst sample is re-verified with the full path (`reverified`).
* **ngspice nondeterminism across versions/builds:** store `simulator_version`; the repro test compares at rel 1e-12 on the same machine only; across machines only the multipliers/values are guaranteed exact.
* **Edge cases:** perturbed `ac_cutoff_3db` leaving the sweep or losing its crossing -> invalid sample (counts as failure; consider noting that as a bound/design issue in the summary); candidates with 0 perturbable parts -> reject with a clear error; `value` rounding to 12 digits changes nothing for E-series values.
* **Default seed 1:** deterministic but invites accidental reuse; replicates need a new seed (documented). Alternative: random seed recorded in the result. I chose fixed for test and docs stability; revisit.
* **MCP client timeouts** are why chunks are 20 s; if a client tolerates longer, `chunk_seconds` up to 60 is allowed.
* **Disk:** a study is about 100-300 KB (jsonl) + the worst-sample artifact (about 27 MB for lab4-class). Consider `keep_worst_artifacts: bool = True` (default true) to drop it.
* **Concurrency:** the non-blocking `session.lock` means an MC chunk makes a concurrent `evaluate_experiment` fail with the standard "operation is active" message for up to `chunk_seconds`; acceptable and documented.
