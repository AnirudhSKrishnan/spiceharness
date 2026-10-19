# SpiceHarness: what it is and where it is going

## The idea in one paragraph

Give SpiceHarness a circuit problem: "I need a gain-of-10 amplifier on ±12 V driving 10 kΩ", or "this supply filter rings; what would work better?". It produces **several candidate circuits that could solve it**, simulates **every one of them in ngspice** under the same frozen test conditions, **ranks them with a deterministic, explainable score**, and then **re-simulates the winner from scratch** to confirm the result. An LLM (Claude Code or Codex) is good at proposing circuits and explaining tradeoffs. It is not trusted to judge them. SPICE measurements decide what passes and what wins.

## Why it exists

Asking an LLM "design me a filter" gets you a plausible circuit with unverified numbers. Running SPICE by hand gets you trustworthy numbers for one circuit at a time. SpiceHarness puts the two together:

| Job | Who does it |
| --- | --- |
| Understand the question, read a textbook figure, suggest topologies and values, explain results | The LLM (your connected Claude Code / Codex session, or optional API agents) |
| Freeze the test bench (supplies, source, load, frequencies, bounds) | The harness, before any candidate is seen |
| Build the netlist, run ngspice, extract measurements | The harness (deterministic Python + real vendor SPICE models) |
| Decide pass/fail and rank the candidates | The harness (fixed lexicographic rule, below) |
| Confirm the winner | The harness: a fresh, independent re-simulation |

The LLM can only submit structured **Circuit IR** (components plus connections from a checksum-pinned catalog). It cannot submit raw SPICE, change the test bench, supply its own measurements, or swap a real device model for an ideal one.

## How a run works

```
 problem ──► frozen spec ──► candidates ──► ngspice ──► measurements ──► rank ──► fresh re-verify ──► report
 (text/fig)  (supplies,      (LLM or       (op, AC,     (gain, cutoff,   (lexi-   (winner rerun      (viewer,
              load, bounds)   templates)    transient)   THD, noise…)     cographic) independently)   KiCad, decks)
```

1. **Freeze the problem.** All electrical conditions go into a spec in SI units: rails, input amplitude (peak) and frequency, source impedance, load, and required bounds such as gain, cutoff, THD, noise and power. Nothing a candidate submits can change them.
2. **Generate candidates.** Either from built-in textbook templates (buffer, inverting/non-inverting amp, passive and active 1st/2nd-order low/high-pass, common-emitter, emitter follower) with E24/E12 values, or as free-form Circuit IR written by the LLM for problems the templates don't cover.
3. **Simulate each one.** The harness runs operating-point, AC and transient analyses, plus optional tolerance corners and injected-noise analysis.
4. **Score and rank.** See the next section.
5. **Re-verify the winner.** A fresh simulation must reproduce a passing result. Only `verified && feasible` counts as success.
6. **Show the evidence.** You get an HTML viewer of every candidate, a terminal dashboard, editable KiCad schematics and the original ngspice decks for every candidate.

## How candidates are ranked

The ranking is a strict priority order, not a weighted blend. A higher rule always beats a lower one:

1. Valid, fully measured circuits come before broken or non-converging ones.
2. **Every circuit that meets all requirements comes before every circuit that doesn't.**
3. Among passing circuits:
   1. fewer physical packages (real parts you'd solder, respecting op-amp channel counts)
   2. then more safety margin on the bounds
   3. then lower power
4. Among failing circuits: the smallest worst-case violation, then the smallest total violation. So the "closest miss" ranks first.

Displayed "points" only illustrate this order; the ordering itself is authoritative. If an experiment has **no requirements**, it is analysis only: nothing is scored, and `feasible` and `rank` stay null. The harness won't invent a winner.

## Two ways to use it

| | Template design (`spiceharness design`, MCP `start_design`) | General experiments (`spiceharness experiment`, MCP `start_experiment`) |
| --- | --- | --- |
| Best for | "Design me an X that meets Y" | "Analyse this circuit from my textbook/lab and compare alternatives" |
| Candidates | Known topologies with standard values; the LLM refines values | Any supported R/L/C plus catalog op-amp/BJT/diode wiring written as IR |
| Example | `examples/amplifier.json`, `examples/family-lowpass.json` | `examples/experiments/lab4-supply/` (power-line filter study) |

Both modes can run with no LLM at all (`--offline`, `experiment run/compare`). They still run real ngspice. They are deterministic benchmarks and are labelled as making no agent calls.

## Try it

```sh
uv sync --python 3.13 && uv run python scripts/fetch_models.py
uv run spiceharness doctor

# Template search: generates, simulates, ranks and verifies candidates (no LLM)
uv run spiceharness design --spec examples/family-lowpass.json --offline

# Compare 8 hand-written supply-filter alternatives under one frozen bench
uv run spiceharness experiment compare --spec examples/experiments/lab4-supply/spec.json \
    --candidates examples/experiments/lab4-supply
uv run spiceharness experiment candidates RUN_ID   # leaderboard
uv run spiceharness view RUN_ID                    # browser view of every candidate
uv run spiceharness export RUN_ID                  # KiCad + ngspice decks for every candidate
```

For the LLM-in-the-loop version, register the MCP server (see README) and ask Claude Code something like: *"Use spiceharness to design a gain-of-10 non-inverting amplifier, ±12 V, 100 mV peak at 1 kHz, 10 kΩ load. Compare candidates, minimise parts, and freshly verify the winner."*

Here is what the Lab 4 compare produces: 8 alternatives, 3 pass, and the simplest passing circuit wins:

| Rank | Candidate | Passes all bounds | Packages |
| --- | --- | --- | --- |
| 1 | `capacitor_4700uf` | yes | 5 |
| 2 | `rc_22ohm_100uf` | yes | 7 |
| 3 | `damped_lc_10mh_100uf` | yes | 11 |
| 4–7 | pi / LC / retuned T / direct | no (closest miss first) | 3–9 |
| 8 | `original_t` (the lab's own filter: strong resonance) | no | 9 |

## Limits

- The result is the **best candidate found within a bounded search** (20 evaluations, 180 s of server work by default). It is not a proof of the global optimum.
- Tolerance corners are a sensitivity screen. They are not an exhaustive worst-case or yield analysis.
- Results are only as faithful as the vendor SPICE models (LM358, OPA1641, LM741, 2N3904/3906, 1N4148). Injected supply noise is not the same as internal device noise or calibrated hardware PSRR.
- Out of scope: arbitrary SPICE devices and control scripts, switching converters, PCB parasitics, and IC-level design.
- Editing an exported KiCad schematic does not carry over the original score. Re-submit the edited circuit and re-verify it.

## Where it should go next

- **Better candidate generation:** let the LLM explore more topologies per problem, and add more templates (Sallen-Key/MFB variants, differential/instrumentation amps, regulators).
- **A larger device catalog**, with the same checksum and provenance rules.
- **Stronger robustness ranking:** Monte Carlo yield, and temperature sweeps as optional ranking criteria.
- **Smoother input:** go from a photo of a textbook circuit to a draft IR with uncertain values flagged for confirmation.
- **Cost/area as optional tie-breakers** after feasibility and part count.

## Current status (October 2026)

**Working end to end:**
- All five template examples (amplifier, low-pass, high-pass, common-emitter, family low-pass) generate, simulate, rank and pass fresh verification.
- The Lab 4 supply-filter comparison ranks 8 alternatives correctly.
- The viewer, KiCad export and exported ngspice decks work.
- Exported ngspice decks rerun successfully from their folders, and the output files match the saved ones in size.
- 200 automated tests pass and lint is clean.

**Fixed in the October 2026 review:**
- **Pasted commands:** the follow-up commands printed after a run (such as `view`) now include the workspace and correct quoting, so they work when pasted.
- **Error messages:**
  - MCP tools now tell the LLM why a request was rejected, instead of a bare "Error executing tool".
  - CLI errors print plainly.
- **Circuits rejected before simulating:**
  - A net named `gnd`. ngspice silently treats it as ground, so the simulated circuit wouldn't match the submitted one.
  - Short-circuit loops (wires, voltage sources and lossless inductors). Before, they wasted an evaluation and failed opaquely in ngspice.
  - Candidate parts that reuse a fixture load's name.
- **Measurements:** a NaN or inf sample anywhere in the simulator output now invalidates the candidate. Before, it could slip through min/max measurements.
- **Finalizing:** finalizing an already-completed experiment with a different or unknown candidate ID is now rejected instead of silently ignored.
- **Verification time:** a search can no longer use up the time reserved for the winner's fresh verification.
- **Cancelling:** cancelling an interrupted (crashed) run now takes effect.
- **Specs rejected up front:** specs the templates can't build are now rejected with a clear message instead of crashing mid-run. These are: an `amplifier` with 0 < gain < 1, absurdly large gains, and `cutoff_hz` on non-filter circuits.
- **Common-emitter search:** mixed amplifier searches explored only half of the common-emitter design points.
- **Knowledge import:** corrupt-PDF imports and empty-file warnings now give clean messages.

**Known open issues:**
- **LM358 on a single supply** (negative rail = 0 V) never reaches a DC solution with the current model and ngspice settings, even in a standalone deck. Those candidates are honestly reported as failed. Dual-supply LM358 and single-supply OPA1641 work.
- **Some active high-pass variants** hit ngspice "timestep too small" in transient analysis and are ranked as failed.
- **Tolerance corners** on circuits with 7 or more passive parts can exceed the 20 s per-evaluation limit.
- **Case sensitivity:** `redesign_ports` and `protected_refs` are case-sensitive. An uppercase port name fails safe (the candidate is rejected) but is confusing.
- **Exit code:** a run that ends `budget_exhausted` still exits with code 0.
- **Late-search timeout:** a search started just before the time reserved for verification gets a very short timeout and may use up an evaluation as `timeout`.
