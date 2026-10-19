# Demo question: filter a 50 kHz interference tone out of an op-amp stage

Paste this into Claude Code or Codex with the `spiceharness` MCP server connected:

> Do what's in @docs/prompts/interference-filter-question.md

---

## Circuit

The same LM741 non-inverting stage as EEET1413 Lab 4:

- **Op-amp:** U1, LM741 on ±12 V rails.
- **Gain:** RF = 2 kΩ (out → inv) and RG = 1 kΩ (inv → ground), so the gain is 3.
- **Input:** a sensor signal of **2 V DC + 1 V peak at 1 kHz**. A **0.2 V peak, 50 kHz** tone, coupled in from a nearby switching regulator, rides on top of it.
- **Source impedance:** 600 Ω.
- **Load:** 10 kΩ.

Without any filter, the interference comes out amplified ×3, at about 0.6 V peak.

## Question

**What is the simplest way to filter the 50 kHz interference while keeping the 1 kHz signal and its DC level intact?**

Compare at least these candidates under one frozen bench, and rank them:

1. no filter (the baseline)
2. a single shunt capacitor at the input, using the 600 Ω source as the R
3. a series-R + shunt-C input filter
4. a capacitor across RF (it makes the stage itself roll off)
5. options 2 + 4 together
6. a two-stage RC-RC input filter
7. an LC input filter, damped enough not to peak

Also try anything else you think could win.

## Acceptance bounds (all must pass)

| Metric | Bound |
| --- | --- |
| AC gain at 1 kHz (from the signal source) | 2.85 – 3.15 |
| Passband peak gain, 10 Hz – 2 kHz | ≤ 3.18 (≈ +0.5 dB) |
| Interference transfer at 50 kHz (from the interference source) | ≤ 0.15 |
| Worst transfer from 50 kHz to 1 MHz | ≤ 0.15 |
| Output mean (DC preserved) | 5.8 – 6.2 V |
| Output 1 kHz amplitude (transient, both tones present) | 2.85 – 3.15 V peak |
| Output 50 kHz amplitude (transient) | ≤ 30 mV peak |
| Output THD at 1 kHz | ≤ 1 % |

Rank by these rules: passing beats failing, then fewer parts, then margin. Freshly verify the winner.

## How to run it

The frozen spec is at `examples/experiments/interference-filter/spec.json`.

**With an agent (client-driven MCP):** pass that spec to `start_experiment`, then submit each candidate with `evaluate_experiment`. Use `get_experiment` to compare, and finish with `finalize_experiment` and `export_run`.

**Without an agent (direct CLI):** use `spice experiment compare --spec examples/experiments/interference-filter/spec.json --candidates <dir of candidate JSONs>`.

## Assumptions

- The 600 Ω source resistance and the 10 kΩ load are fixed and belong to the test bench.
- The ±12 V rails are ideal. Supply noise is a different question; see `lab4-noise-filter-question.md`.
- U1 is fixed to the LM741 vendor model. RF and RG and any added passives may change.
- AC transfer is measured from each source alone. The transient analysis runs with both tones present.

## What theory predicts

- **A first-order filter can't pass.** Keeping ≥ 0.95 of the signal at 1 kHz forces the cutoff to fc ≥ 3 kHz. At 50 kHz that leaves about 0.06 of the input, which comes out as ×3 ≈ 0.18 and fails the 0.15 bound.
- **Two poles are needed.** The cheapest way to get them is probably to reuse the source resistance with one shunt capacitor, plus a capacitor across RF. That is 2 added parts.
- **An LC filter is also 2 parts.** It needs damping (Q ≲ 0.7) to stay under the passband-peak bound.

---

## What SpiceHarness found (run `0a618bcf5a6b40d8a8394235800be4fa`, 19 Oct 2026)

Seven candidates were submitted through the client-driven MCP flow (`start_experiment`, then `evaluate_experiment`, then `finalize_experiment`). Each was simulated in ngspice with the LM741 vendor model under the frozen bench. The candidate JSONs are in `examples/experiments/interference-filter/`.

| Rank | Candidate | Passes all bounds | Parts | 1 kHz gain | 50 kHz transfer | 50 kHz at output |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | `shunt_47nf_feedback_12nf`: 47 nF input → ground, 12 nF across RF | **yes** | 5 | 2.92 | 0.118 | 23.5 mV |
| 2 | `lc_8m2h_47nf`: 8.2 mH (20 Ω ESR) series, 47 nF shunt | **yes** | 5 | 3.00 | 0.078 | 15.4 mV |
| 3 | `rc_rc_47nf_1k_15nf`: 47 nF, then 1 kΩ, then 15 nF | **yes** | 6 | 2.89 | 0.068 | 13.4 mV |
| 4 | `rc_1k_33nf`: series 1 kΩ, shunt 33 nF | no (50 kHz 0.179) | 5 | 2.85 | 0.179 | 35.7 mV |
| 5 | `shunt_c_82nf`: one 82 nF cap, using the 600 Ω source | no (50 kHz 0.192) | 4 | 2.87 | 0.192 | 38.3 mV |
| 6 | `feedback_cap_12nf`: 12 nF across RF only | no (50 kHz 1.05) | 4 | 2.97 | 1.05 | 210 mV |
| 7 | `no_filter` (baseline) | no | 3 | 3.00 | 2.98 | 593 mV |

The winner was **freshly re-simulated** and reproduced `verified = true` and `feasible = true`.

**Answer.** Two small capacitors are enough:

- one from the op-amp + input to ground, which forms an RC with the 600 Ω source
- one across the feedback resistor

That gives two poles (−3 dB at about 4 kHz). It cuts the 50 kHz tone from 593 mV to 23.5 mV and keeps the 1 kHz gain at 2.92 and the 6.0 V DC level.

As theory predicted, every first-order option (rows 4 to 6) misses. A cap across RF alone can never go below a gain of 1, because the stage is non-inverting.

**Caveats:**

- **Tie-break.** Rows 1 and 2 tie on the ranking key: both pass, both use 5 parts. The general-experiment rank key has no margin tie-break, so the earlier submission is listed first. The LC filter has more margin, a flatter passband and better rejection, but needs an 8.2 mH inductor. The two-cap version is cheaper and smaller. Pick by what matters to you.
- **Not a worst-case proof.** All values are nominal. A tolerance (Monte Carlo) check is the planned next step.
