# spiceharness

A local analog circuit investigation and design harness built with ngspice, MCP, and optional LangGraph API orchestration. Your connected Codex or Claude Code interprets questions and circuit images, submits supported circuit IR, compares measured evidence under frozen operating conditions, and explains the result. A local browser viewer shows every saved candidate circuit, measurements, score checks and artifacts; a live terminal dashboard shows progress. Required electrical checks always dominate physical component count.

The connected client workflow needs no separate model API keys. Optional API orchestration still supports OpenAI and Anthropic independently for designer, scorer, and expert roles, including mixed configurations. Agents explain results; deterministic Python evaluators own measurements, feasibility, and ranking.

## Start here

Python 3.13+, [uv](https://docs.astral.sh/uv/), and [ngspice](https://ngspice.sourceforge.io/) are required. On macOS:

```sh
brew install ngspice
uv sync --python 3.13
uv run python scripts/fetch_models.py
uv run spiceharness doctor --simulate-models
uv run spiceharness design --spec examples/amplifier.json --offline
```

`--offline` runs an explicitly deterministic template benchmark with **no model calls**. It still runs real ngspice, evaluates candidates, verifies the finalist, and produces artifacts. It is useful before configuring credentials. Passive RC examples do not need external device models.

```sh
uv run spiceharness design --spec examples/lowpass.json --offline
uv run spiceharness design --spec examples/transistor.json --offline
uv run spiceharness design --spec examples/highpass.json --offline
```

Vendor models are downloaded into the local source tree, not redistributed with the harness. Downloads are pinned by SHA-256; unexpected changes are rejected. Original model headers and source references are retained. See [model sources](src/spiceharness/data/models/README.md). Installation is complete only after the model fetch succeeds.

## Autonomous agents: OpenAI or Claude

Set the credentials required by your configuration in the environment. The CLI does not automatically read `.env` files. Keep credentials out of specification files and source control.

```sh
# After setting OPENAI_API_KEY in your shell:
uv run spiceharness design --spec examples/amplifier.json --config configs/openai.toml

# After setting ANTHROPIC_API_KEY in your shell:
uv run spiceharness design --spec examples/amplifier.json --config configs/claude.toml

# After setting both credentials:
uv run spiceharness design --spec examples/amplifier.json --config configs/mixed.toml
```

The OpenAI preset uses GPT-6.1 Sol for designer/scorer and GPT-6 Astra at high reasoning for expert consultation. The Claude preset uses Claude Sonnet for designer/scorer and Claude Opus for the expert. Edit each role's provider, exact model ID, reasoning effort, and output-token limit in TOML. Model availability is checked against your account before autonomous work; there is no silent provider fallback. MCP can also inherit the configuration through `SPICEHARNESS_CONFIG`.

The default budget is 180 seconds, 20 circuit evaluations including a reserved fresh verification, one expert consultation, and 24 API requests including retries and access checks. Limits are configurable in TOML. API timeouts are bounded by the run deadline. Tokens and API calls are recorded; these limits are not a dollar-cost guarantee.

Natural-language input uses the configured designer to extract the stated constraints. Missing required conditions are reported before a run starts:

```sh
uv run spiceharness design --config configs/claude.toml --request "A noninverting gain-of-10 amplifier, plus and minus 12 V supplies, 100 mV peak input at 1 kHz, 10 kilohm load."
```

## Circuit specifications

Examples use a strict JSON `DesignSpec`. All numeric quantities use SI units: volts, hertz, ohms, amperes. Input amplitude is **peak**, not peak-to-peak. Supply rails, input amplitude/frequency, and load must be explicit. The evaluator owns the sources, source impedance, and load; candidate agents cannot change them.

Supported seed topologies:

- `buffer`, `inverting_amplifier`, `noninverting_amplifier`
- `passive_lowpass`, `passive_highpass`
- `active_lowpass_1`, `active_highpass_1`, `active_lowpass_2`, `active_highpass_2`
- `common_emitter`, `emitter_follower`

Use `"topology": "amplifier"`, `"lowpass"`, or `"highpass"` to compare compatible textbook topologies under the same frozen specification. See [family-lowpass.json](examples/family-lowpass.json) for a gain/cutoff/stopband request that compares passive and active alternatives. Explicit topology names restrict the search to that circuit family. Resistors use E24 values and capacitors use E12 values. Device choices come from a checksum-pinned catalog: LM358, OPA1641, 2N3904, 2N3906, and 1N4148. An unavailable or incompatible device model causes an explicit failure, never an idealized replacement.

The catalog enforces its declared supply and temperature envelopes. These checks do not replace complete component stress, thermal, or tolerance analysis. Inverting amplifiers and common-emitter stages require a negative signed gain; negative-gain filter recipes are unsupported. `min_filter_order: 2` requests a second-order family. A normal run performs operating-point, AC, and transient analyses. It measures loaded gain, phase, cutoff where applicable, DC offset, current, distortion, output swing, and gain compression. A failed or nonconvergent analysis, missing required measurement, clipping, or changed specification cannot pass. Optional `corner_policy: "tolerance"` adds four correlated R/C cases plus individual positive/negative perturbations of each resistor and capacitor. These sensitivity screens expose ratio errors but are not exhaustive worst-case or yield analysis. `supply_power_w` is signed net DC power delivered by the declared rails, clamped at zero; `supply_current_a` is that power divided by the rail span. `total_rail_current_a` separately records the sum of absolute rail currents. Device and temperature variation are only as faithful as the selected SPICE models.

## How the agents and score work

The LangGraph workflow retrieves references, generates an analytical pool of known circuit families and standard component values, runs simulations, scores results, and independently reruns the finalist. Designer and expert agents select and reprioritize eligible candidates using measured feedback; the scorer agent reviews the numeric evidence. The expert is consulted after repeated lack of improvement. This template workflow selects from declared topology skeletons. The general experiment workflow below also accepts supported circuit connections supplied by the connected client.

The authoritative ranking is lexicographic:

1. Valid, completely measured candidates before invalid candidates.
2. Every passing design before every failing design.
3. Passing designs: fewer physical packages, then greater capped margin, then lower power.
4. Failing designs: smaller worst normalized violation, then smaller total violation.

The displayed points break down feasibility, simplicity, margin, and power. Ranking remains authoritative. Bias, feedback, coupling, and required decoupling parts count. Bench sources and declared loads do not. Op-amp internal model elements do not count as physical parts; package channel limits prevent hiding multiple devices under one package.

The result is the **best design found within the budget**, not proof of a global minimum. Arbitrary SPICE devices/control scripts, switching-converter design automation, PCB extraction, and transistor-level IC process design are outside the supported workflows. General experiments support declared R/L/C and catalog-device connections, explicit lumped series losses, and bounded analyses; they do not imply support for every circuit or analysis.

## Inspect, resume, cancel, and export

```sh
uv run spiceharness design --spec examples/lowpass.json --offline --detach
uv run spiceharness show RUN_ID
uv run spiceharness show RUN_ID --watch
uv run spiceharness view                # open every candidate circuit and score for the latest run
uv run spiceharness view RUN_ID -w /path/to/workspace
uv run spiceharness view RUN_ID --no-open  # export HTML without launching a browser
uv run spiceharness export RUN_ID         # ALL candidates as native KiCad + ngspice files
uv run spiceharness export RUN_ID --candidate UNIQUE_ID_PREFIX --output /path/to/circuits
uv run spiceharness dashboard           # watch the latest saved run
uv run spiceharness dashboard RUN_ID -w /path/to/workspace
uv run spiceharness show RUN_ID --json
uv run spiceharness cancel RUN_ID
uv run spiceharness resume RUN_ID
```

### Short `spice` commands from any directory

`scripts/install_shortcuts.sh` writes `~/.local/bin/spice`, a small wrapper around this checkout's `.venv` (Homebrew's ngspice on `PATH`, default workspace baked in). It is idempotent, refuses to overwrite a file that is not a SpiceHarness wrapper, and supports `--workspace DIR`, `--bin-dir DIR` and `--uninstall`. `spice` is also a second console script of the same app, so every `spiceharness` subcommand works as `spice ...`.

```sh
scripts/install_shortcuts.sh            # once; default workspace = this checkout
spice runs                              # recent design + experiment runs (read-only; --limit N, --json)
spice dashboard                         # latest run in the terminal
spice view [RUN] [--no-open]            # browser viewer
spice candidates [RUN]                  # leaderboard (design or experiment)
spice candidate RUN CANDIDATE           # one circuit with its score and artifacts
spice show [RUN]                        # full saved summary (--json, --watch)
spice export [RUN]                      # KiCad + ngspice files for every candidate
```

`RUN` and `CANDIDATE` accept a unique prefix (RUN: at least 4 hex characters); an ambiguous prefix is an error. Without `-w`, the workspace is the first of: `SPICEHARNESS_WORKSPACE`; the current directory if it contains `.spiceharness/`; `SPICEHARNESS_DEFAULT_WORKSPACE` (set by the wrapper unless already set); the current directory. `-w/--workspace` always wins. Printed follow-up commands use `spice ...` when the wrapper is installed, adding `-w` only for a non-default workspace.

Runs are persisted under `.spiceharness/` in the selected `--workspace`. Background runs survive client disconnection. Saved run data includes the frozen specification, circuit IR, candidate results, leaderboard, model checksums, sources, provider usage, and checkpoints. Artifacts include SPICE decks/raw data/logs, the finalist BOM, schematic, and report. A stopped run retains completed work. Resume preserves its search history and remaining budget.

On interactive terminals, Rich refreshes the schematic and leaderboard. Redirected output uses a static view; `--json` provides machine-readable output. Schematics use shared circuit connectivity and include a connection table where a reliable compact drawing is unavailable.

`dashboard` opens a terminal view of an existing run and never starts a new design. Omitting the ID selects the latest run in the selected workspace. Completed runs display their saved final results.

`export [RUN_ID] [-w WORKSPACE]` writes a portable folder under `.spiceharness/exports/RUN_ID` by default. Each candidate, including pending, failed and fresh verification records, has an editable `candidate.kicad_sch`, original Circuit IR, saved score/metrics/evidence, and available original `.cir`, `.raw`, `.log`, CSV and NPZ files. The manifest maps every candidate's ID, name, purpose, saved rank, feasibility and verification to its folder; the root README points to the fresh finalist schematic. Analysis-only runs retain null ranks and have no claimed winner. `--candidate` selects an exact ID or unique prefix. Existing exports and user edits are preserved: subsequent exports create `RUN_ID-2`, `RUN_ID-3`, and so on.

Open `candidate.kicad_sch` in KiCad Schematic Editor. The native drawing includes the frozen bench fixture and catalog pin mappings. Model files are copied locally only when their hashes match both the saved evaluator checksums and the trusted source catalog; no models are downloaded and no ideal substitute is created. Missing original models are reported and make that circuit unavailable for simulation. Existing decks are checked against the validated IR and frozen fixture before export, retain their filenames and relative `model_N.lib` includes, and run after moving the whole candidate folder. From inside that folder, use the specific filename listed in its README, for example `HOME="$PWD" SPICE_ASCIIRAWFILE=1 ngspice -b -r user-output.raw -o user-output.log 0_op.cir`. The command scopes ngspice's PSpice compatibility initializer to the exported folder and leaves your personal configuration alone. KiCad's interactive Simulator uses its own selected analysis settings; the original evaluator decks remain the reproduction reference. Export calls no agents or simulator, preserves all saved electrical results, and includes only regular local artifact files, never credentials or provider settings.

Saved scores apply only to the original evaluated circuit. After manually editing a KiCad schematic, submit validated Circuit IR as a new candidate, re-evaluate it under the frozen specification, and require fresh verification before claiming it passes; the old score does not establish validity of the edited circuit. Vendor models are copied for local use with source/checksum provenance. The export does not grant permission to redistribute those files.

## Reference knowledge

The starter knowledge base contains original engineering recipe notes with citations to [Lessons in Electric Circuits](https://www.allaboutcircuits.com/textbook/). It does not claim to contain or have trained on commercial textbooks. Retrieval supplies context; SPICE establishes measured feasibility.

Import your local reference material:

```sh
uv run spiceharness knowledge import /path/to/analog-notes.md
uv run spiceharness knowledge import /path/to/textbook.pdf
uv run spiceharness knowledge search "low pass source load"
```

Imports support UTF-8 text, Markdown, and text-based PDFs up to 25 MiB. PDF chunks retain page numbers and document checksums. Scanned pages are flagged for external OCR; equations and figures are not reconstructed from images. Retrieved text is treated as untrusted reference data. Fine-tuning is not part of this version.

## MCP: use Codex or Claude Code without separate API keys

In the default MCP mode, **your connected coding agent supplies the LLM**. SpiceHarness makes no OpenAI or Anthropic API calls: it supplies circuit proposals, validates component edits, runs ngspice, awards deterministic scores, and verifies the finalist. You use the normal sign-in and model settings of Codex or Claude Code. Independently configured API agents remain an explicit optional mode.

After `uv sync --python 3.13` and the model download, register the local stdio server:

```sh
# Codex (user configuration):
codex mcp add spiceharness -- /path/to/spiceharness/scripts/serve_mcp.sh

# Claude Code (private registration for this project):
cd /path/to/spiceharness
claude mcp add --scope local --transport stdio spiceharness -- /path/to/spiceharness/scripts/serve_mcp.sh
```

Replace the absolute repository path on another computer. The launcher resolves the Python environment and common Homebrew binary paths, and reserves stdout for MCP. Registration uses the clients' supported commands: [Codex MCP documentation](https://learn.chatgpt.com/docs/extend/mcp?surface=cli), [Claude Code MCP documentation](https://code.claude.com/docs/en/mcp).

Open a new Codex or Claude Code session in this project. Check `/mcp`, then ask:

> Use spiceharness to design a gain-of-10 noninverting amplifier with +12 V and -12 V supplies, a 100 mV peak input at 1 kHz, and a 10 kilohm load. Compare candidates, minimize physical components, and freshly verify the winner. Use client mode; do not request API keys.

The connected agent follows this sequence:

1. `start_design(spec)` freezes the electrical conditions and returns a persistent `run_id` in `awaiting_client` state.
2. `propose_candidates(run_id)` returns eligible textbook topologies and component values.
3. `evaluate_candidate(run_id, circuit)` measures a proposal and returns the server-calculated points and ranking. The agent can refine E24 resistor and E12 capacitor values while preserving the supported topology, real devices, and frozen specification.
4. `get_run` and `score_candidate` inspect saved evidence. `render_circuit` shows the schematic and BOM; `search_knowledge` supplies cited notes.
5. `finalize_design(run_id)` independently reruns the numerical winner and saves the circuit, schematic, BOM, and report. Both `verified` and `feasible` must be true to claim success.

If an analysis fails, inspect the saved errors and try other eligible candidates within the run budget. Preserve the real device models and frozen conditions; if none passes fresh verification and all constraints, report that no verified, feasible design was found.

The server also exposes the `spiceharness://guide` resource and `design_circuit` MCP prompt. Runs survive client disconnection: reconnect to the same workspace and continue with the saved ID. `cancel_run` remains available while a SPICE operation is active.

MCP runs in the background and opens no window automatically. The host agent should show the returned `run_id`, `viewer_command`, and `dashboard_command`, plus the evaluated candidate count, selected circuit name/ID, saved score and failed checks after evaluations and finalization. Paste `viewer_command` into a terminal to open a browser snapshot of every candidate circuit and its evidence. It uses absolute interpreter and workspace paths, so it works from another directory. From this project run `uv run spiceharness view [RUN_ID] [-w WORKSPACE]`; omitting the ID selects the latest saved template run or general experiment. `--no-open` writes `.spiceharness/{runs|experiments}/RUN_ID/report/results.html` and prints its absolute path and file URL without opening a browser. Re-run the command to refresh the snapshot. A browser launch failure retains the artifact and returns a nonzero status with the URL. `dashboard` remains the terminal progress interface; completed runs print a full static summary without terminal-height clipping.

For textbook questions and supported circuits beyond the templates, read `spiceharness://experiments` or use the `investigate_circuit(question)` prompt. The host interprets supplied images and documents, retains source/page provenance, resolves uncertain values and connections, and explains cited theory. The harness accepts structured circuit IR rather than image files, textbook prose, arbitrary netlists, or SPICE measurement expressions.

The general experiment lifecycle is:

1. `start_experiment(spec)` freezes `ExperimentSpec`: voltage/current sources, source impedance expressed by fixed loads or protected circuit parts, resistive loads, named differential probes, analyses, typed measurements, optional metric bounds, allowed device models, baseline, protected component references, redesign ports, assumptions, and source/page provenance.
2. `evaluate_experiment(run_id, circuit)` measures the baseline, then each alternative as `GeneralCircuitIR` under that identical fixture. Protect the host circuit and compare alternatives at the same supplies, signal, source impedance, load, bandwidth, and temperature. The server rejects changed protected parts, unknown fields, raw controls, and supplied scores or measurements.
3. `get_experiment(run_id)` returns saved measurements and the deterministic leaderboard. All required bounds dominate component count; among failing candidates the worst and total normalized violations precede packages. With no `requirements`, the run is analysis only: `feasible` and `rank_key` are null, and there is no numerical feasibility ranking.
4. `finalize_experiment(run_id)` freshly simulates the deterministic winner and saves the frozen fixture, circuit IR, measurements, model provenance, decks/logs/raw data, sampled curves, and a report. In analysis-only mode, an optional saved `candidate_id` selects the evidence to remeasure. A valid fresh simulation sets `verified`; separately inspect `feasible` before claiming an optimization meets its bounds. `completed` alone is insufficient.
5. Explain the baseline/alternative comparison, cite the source reasoning, identify discrepancies from theory, and state the demonstrated conditions. `cancel_experiment` remains available during simulation and final report preparation. An interrupted pending evaluation retains its consumed budget and never becomes measured evidence.

Supported primitives are R/L/C, catalog op-amps, BJTs and diodes, ideal connections, and intrinsic lumped R for reactive components. Explicit external resistors count as packages; intrinsic loss belongs to the existing physical part. Sources support DC, sine, and pulse waveforms. Typed measurements include operating-point voltage/source current, AC magnitude and band extrema, interpolated relative 3 dB crossings, transient mean/extrema/fundamental/THD, and externally injected noise RMS. Bounds and sample/work limits are validated before simulation. This is a bounded general circuit interface, not unrestricted SPICE.

External noise uses declared independent one-sided source PSDs in V²/Hz or A²/Hz and measured transfer functions integrated over an explicit bandwidth. Contributions sum in power. Rail injection can measure supply-to-output transfer under the declared source impedance and operating point; it does not establish internal device noise, correlated noise, calibrated hardware PSRR, or exact equivalence to another simulator's noise-percentage convention. Nominal sampled results and sensitivity screens are not exhaustive worst-case proof.

Inspect all general candidate names, IDs, saved ranks, package counts, feasibility and maximum violations with `uv run spiceharness experiment candidates RUN_ID`. Inspect an individual circuit's exact connections (including ideal wires), measurements, per-bound checks and generated `.cir` netlist paths with `uv run spiceharness experiment candidate RUN_ID CANDIDATE_ID_OR_UNIQUE_PREFIX`. Add `-w WORKSPACE` when needed. Netlists are saved simulator inputs, not an application-native drawn schematic.

General experiments do not award points: scored comparisons display the evaluator's saved `rank_key`, normalized per-bound violations and physical package count. All required bounds dominate package count. If `requirements` is empty, the run is **not scored (analysis only, no acceptance requirements)**: feasibility and rank are null, and a selected freshly measured circuit is not a numerical winner. To compare against acceptance criteria, start a new frozen experiment with explicit bounds; never retrofit scores onto the original evidence.

Each general session permits 20 evaluations including one reserved fresh verification, a 180-second server-work budget, and a 20-second evaluator timeout. Client thinking/disconnection time is excluded. Completed evaluation/postprocessing time is charged at its actual duration; crash recovery can only estimate interrupted work up to one operation allowance. The dashboard selects the latest saved template run or general experiment and never opens a window automatically.

The direct CLI driver reads a frozen spec and supplied circuit JSON files, runs real SPICE, and freshly verifies the winner. It performs **no agent calls** and is labeled a deterministic benchmark; it does not generate alternatives or interpret the question:

```sh
uv run spiceharness experiment run --spec examples/experiments/loaded-rc/spec.json --candidate examples/experiments/loaded-rc/reference.json
uv run spiceharness experiment compare --spec examples/experiments/lab4-supply/spec.json --candidates examples/experiments/lab4-supply
uv run spiceharness experiment show RUN_ID --json
uv run spiceharness dashboard RUN_ID
```

`compare` selects circuit-shaped JSON files in the directory, excludes `spec.json` and `*-spec.json`, accepts at most 19 candidates, and reserves its last evaluation for fresh verification. See [the Lab 4 acceptance study](docs/lab4-study.md) for explicit interpretation, real-model measurements, and conditional tradeoffs. Both commands support `--workspace` and `--json`. General experiments are persisted under `.spiceharness/experiments/`.

If new tools are missing after an update, restart the coding client to reload the MCP tool definitions. If a local write tool such as `start_design` needs approval, approve that requested tool in a normal interactive client session. A client configured to forbid approvals may block the workflow before a run starts. Read-only MCP connectivity and SDK tests do not establish that the full signed-in client workflow can run; client mode still needs no separate model API keys.

Client mode defaults to at most 20 circuit evaluations including final verification, with 180 seconds of server operation time; client thinking and disconnected time are excluded. Search reserves evaluation count and time for verification. Each SPICE operation is bounded to 20 seconds. The server rejects supplied measurements, altered fixtures, unsupported wiring, and model substitutions.

`start_design(spec, execution_mode="autonomous")` explicitly runs the separate API-backed LangGraph agents and needs the provider keys described above. `offline=True` selects the separate deterministic benchmark. Default client mode calls neither model APIs nor the automatic offline search.

For other MCP clients, use [examples/mcp.json](examples/mcp.json) or run `uv run spiceharness serve --workspace /absolute/path/to/spiceharness`. Tools do not accept arbitrary shell commands or raw SPICE control blocks.

## Credentials and local files

API keys are only needed for the optional autonomous API mode. Set `OPENAI_API_KEY` and/or `ANTHROPIC_API_KEY` in your shell, never in specs, configs or source files. `.env.example` lists the variable names. A local `.env` is git-ignored, though the CLI does not load it automatically.

These are also git-ignored and never pushed: `.spiceharness/` (local run databases, logs and exports) and downloaded vendor `.lib` models.

## Development and validation

```sh
uv run pytest -q
uv run ruff check src tests scripts
uv build
```

Tests use real ngspice when installed, pinned device models when downloaded, and mocked provider responses. General experiment checks include closed-form RC results, manufacturer-model measurements, actual SDK stdio lifecycle/reconnect without keys, analysis-only behavior, protected-fixture tamper rejection, independent fresh verification, concurrent cancellation, crash recovery, reserved budgets, and direct CLI/dashboard routing. Live provider checks require configured credentials; the offline tests do not establish API-account access. Benchmark examples cover amplifier gain, filter cutoff, and transistor bias. Tests also exercise infeasible candidates, invalid metrics, package counting, interface lifecycle, provider output validation, and persistence.

Framework references: [LangGraph](https://docs.langchain.com/oss/python/langgraph/overview), [MCP Python SDK](https://py.sdk.modelcontextprotocol.io/), [ngspice batch execution](https://ngspice.sourceforge.io/ngspice-control-language-tutorial.html), [OpenAI model API](https://developers.openai.com/api/docs/guides/latest-model?model=gpt-6.1-sol), and [Claude model API](https://platform.claude.com/docs/en/models/overview).
