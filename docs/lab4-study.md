# Lab 4 acceptance example

Run from the repository root:

```sh
uv sync --python 3.13
uv run python scripts/fetch_models.py LM741
uv run python scripts/run_lab4_study.py
```

This is a deterministic script calling no agents or provider APIs. It exercises the same generic `ExperimentManager.start_experiment`, `evaluate_experiment`, `finalize_experiment`, and `get_experiment` API available to connected clients. All decks come from validated Circuit IR. The script does not add lab-specific behavior to the engine. Matplotlib is a development dependency for the example's plots.

The command prints the comparison run ID, report path and `uv run spiceharness dashboard RUN_ID`. The local report includes PNG/SVG Bode plots, nominal input/output oscilloscope traces, integrated external-noise bars, a complete metric CSV, model hashes, and evidence paths. Local model copies, simulator outputs and reports remain untracked under `.spiceharness/`. A separate analysis-only session measures the original T bandwidth; another session independently checks an ordinary loaded RC circuit.

## Source and frozen interpretation

The source is the user-supplied `EEET1413_Lab4NOTE (1).docx`, Task 1 Q2 Figure 1 and Task 2 Figure 2, SHA-256 `5839a01e18f91c8d52189aab6039d2d566fb625496bfe24c79a6c246acccd79e`. The other local copy has the same hash. The source is neither edited nor distributed. Its figure uses a 741; the fixture explicitly chooses the original TI LM741 model, fetched with archive/file checksums through `scripts/fetch_models.py`.

[The fixture](../examples/experiments/lab4-supply/spec.json) freezes RF=2 kΩ, RG=1 kΩ, ±12 V rails, and input 2 V DC + 1 V peak at 1 kHz. The literal T on **each** rail is first series 7.95 mH, middle shunt 6.4 nF, then series 7.95 µH. No extra 100 nF bypass is inserted into the baseline or literal T. TI recommends practical local supply bypassing; adding it changes the circuit and must be counted as an alternative [1].

Missing source/load details become explicit assumptions: 1 Ω source impedance per rail, ideal signal source, 10 kΩ output load, 27 °C; original inductor DCR 1 Ω each, alternative DCR 0.05 Ω, capacitor ESR 0.1 Ω. These losses are part attributes, each capacitor or inductor still counts as one package. All candidates retain the protected amplifier. Counts include the three fixed amplifier packages plus every filter package; bench impedances/load are fixed test equipment.

The conditional external-noise case is independent 10 mV RMS white noise on each rail over 10 Hz–100 kHz. Its one-sided PSD is `(0.01 V)^2 / (100000−10) Hz`. Each source's complex AC transfer is measured separately; variances add as `∫Σ|H_i(f)|²S_i(f)df`. This is external injected noise, not internal resistor/op-amp noise. Transient plots show the actual nominal sine simulation; no invented random waveform is added.

The source's thermal-noise block records 10 MΩ, 27 °C, 10 GHz and “ratio 100.” The ideal resistor relation `sqrt(4kTRB)` gives about 40.7 mV RMS [4]. Multiplying amplitude by 100 gives 4.07 V; multiplying noise power by 100 gives 0.407 V. The ratio's meaning and instrument loading are unresolved. The fixture does not claim either as exact Multisim equivalence. A lumped 741/filter model is not established as valid at 10 GHz.

## Q1: purpose and alternatives

A low-pass T network places two series impedances around a shunt branch. Inductors impede changing current while the shunt capacitor returns AC current; practical losses and terminations determine attenuation and resonance. Input-filter resonance needs damping and interaction checks [2]. These original recipe notes are cited design reasoning, not a claim of textbook training.

| Network | Useful feature | Tradeoff to check |
| --- | --- | --- |
| L–C–L T | Two series isolation elements | Inductor size, DCR, resonance and load dependence |
| RC | Simple, damped, inexpensive topology | DC voltage drop and dissipation |
| LC | Low DC loss, strong roll-off | Potentially large resonant peak |
| C–L–C π | Extra source-side shunting | Extra capacitor and resonant behavior |
| Ferrite plus capacitor | High-frequency loss can help suppression | Bias dependence and bead/capacitor resonance [3] |
| Local bulk capacitor | Few components when source impedance helps | Large capacitance, inrush, ESR floor and size |

Sensitive converters and RF circuits are examples where rail noise matters [3]. Sensor amplifiers are another application: finite supply rejection can allow rail disturbance to contaminate small signals [5].

## Q2: amplifier and measured output

This is a non-inverting amplifier. Ideal closed-loop gain is `1+RF/RG=3`, giving `6 V + 3 V sin(2π1000t)`: approximately 3–9 V. The real LM741 baseline simulation gives gain **3.00004**, DC output **6.00330 V**, nominal transient **3.00329–9.00331 V**, and fitted fundamental **3.00002 V peak**. Those nominal sine traces cannot establish output random-noise amplitude. Switching supply edges and changing shared-rail load currents are two noise mechanisms; common impedance and electromagnetic coupling make layout relevant [6].

## Q4 and Q5: bandwidth and comparison

The loaded literal T's falling −3 dB crossing relative to its 10 Hz response is **34.419 kHz**. Its sampled positive-rail peak is **201.73 V/V**. Calling this simply a clean “34 kHz low-pass bandwidth” would hide the resonance. The ideal natural-frequency estimate `1/(2πsqrt(7.95mH·6.4nF))≈22.3 kHz` is not a loaded −3 dB bandwidth.

The frozen engineering checks require gain 2.85–3.15, filter DC drop ≤0.1 V per rail, worst sampled rail transfer ≤0.1 over 1–100 kHz, peak ≤+3 dB over 10 Hz–100 kHz, valid rail/headroom and nominal 3–9 V swing/low distortion. See JSON for exact bounds; every required check dominates package count. These are declared investigation requirements, not claimed verbatim lab requirements.

| Candidate | Packages | Required checks | Positive-rail integrated noise |
| --- | ---: | --- | ---: |
| Direct baseline | 3 | Fail attenuation | 10.000 mV RMS |
| Literal T | 9 | Fail attenuation/peaking | 138.98 mV RMS |
| RC 22 Ω / 100 µF | 7 | Pass | 0.3164 mV RMS |
| LC 10 mH / 100 µF | 7 | Fail peaking | 1.4704 mV RMS |
| Damped LC | 11 | Pass | 0.4636 mV RMS |
| π | 9 | Fail peaking | 1.4686 mV RMS |
| Retuned T | 9 | Fail peaking | 1.4704 mV RMS |
| 4700 µF per rail | 5 | Pass | 0.9299 mV RMS |

The literal T worsens this particular broadband case; adding reactive components does not guarantee improvement. Nominal waveform gain remains near three with and without it. Output supply-noise predictions are respectively **1078 µV RMS** baseline, **6603 µV** literal T, **4.84 µV** RC, **0.148 µV** damped LC, and **98.0 µV** bulk capacitor. They are **unvalidated macromodel coupling predictions**, not established real LM741 AC PSRR or measured hardware noise. Output noise is reported without a pass bound because that model behavior has not been validated. TI's datasheet DC rejection specification does not validate this frequency-dependent model [1].

The fresh deterministic winner is the bulk-capacitor candidate: five packages among these eight submissions. RC attenuates more but uses seven packages and drops 91.3 mV on the positive rail. Damped LC uses eleven packages. This is not a global optimum or a production recommendation: price, volume, voltage/current ratings, inrush, leakage, capacitor ESL, layout, tolerances and startup are unconstrained. The 4700 µF result specifically depends on the assumed 1 Ω source. No sensitivity screening or exhaustive worst-case analysis is claimed. The 200-point/decade grid can miss a sharper resonance peak.

## Q6 and the live demonstration

Two further techniques are a suitable low-noise regulator/post-regulator with adequate headroom and frequency-specific rejection, and careful return routing/short local decoupling loops to reduce shared impedance and coupling. Both need validation in the actual assembly [1], [5]. Additional practical bypass parts count toward package totals in any alternative that uses them.

Task 3 explicitly requires an in-person NI Multisim demonstration and NI simulation files. This ngspice evidence does not satisfy that task or generate a Multisim submission. The original source document remains unchanged.

## Ordinary loaded RC oracle

[The independent example](../examples/experiments/loaded-rc/spec.json) uses a 50 Ω source, series 1 kΩ, shunt 100 nF with 0.2 Ω ESR, and 10 kΩ load. With `Zc=0.2+1/(j2πf·100nF)`, `Zp=10kΩ || Zc`, its exact loaded transfer is `Zp/(1050Ω+Zp)`. The script compares complex transfer against **every** saved ngspice AC point and requires max absolute error ≤1e−7; the recorded run gave **1.21e−12**. Its approximate pole is 1.675 kHz. This example needs no device model and demonstrates that the engine is reusable beyond the lab.

## References

[1] Texas Instruments, “LM741 Operational Amplifier,” SNOSC25D, Oct. 2015, pp. 4–5, 9–11. [Datasheet](https://www.ti.com/lit/ds/symlink/lm741.pdf).

[2] Texas Instruments, “Input Filter Design for Switching Power Supplies,” SNVA538, pp. 2–4. [Application report](https://www.ti.com/lit/pdf/snva538).

[3] J. Jefferson Eco and A. Limjoco, “Ferrite Bead Demystified,” AN-1368, Analog Devices. [Application note](https://www.analog.com/en/resources/app-notes/an-1368.html).

[4] Analog Devices, “Activity: Noise,” resistor thermal-noise relation. [Educational laboratory](https://wiki.analog.com/university/labs/noise_adalm2000).

[5] T. Owen and A. Patel, “Measuring 2nV/√Hz Noise and 120dB Supply Rejection on Linear Regulators,” AN-159, Analog Devices. [Application note](https://www.analog.com/en/resources/app-notes/an-159.html).

[6] Texas Instruments, “Reducing noise in a buck power-supply design,” SSZTBF4. [Technical article](https://www.ti.com/document-viewer/lit/html/SSZTBF4).
