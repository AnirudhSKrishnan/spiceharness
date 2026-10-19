# Device model sources

`catalog.json` pins original device model bytes, source URLs, archive members, model names and canonical pin order. Run `uv run python scripts/fetch_models.py` from the repository to populate the local model directory. Existing files are verified before reuse; mismatched files are rejected.

Vendor `.lib` files are downloaded locally and ignored by Git. Their original headers are retained. They are separate from the distributed harness source code. The source suppliers' terms and model limitations apply. No model is silently replaced with an ideal device.

| Catalog device | Source | Model identifier | Notes |
| --- | --- | --- | --- |
| LM358 | Texas Instruments, SNOM268 | LMX58_LM2904 | Model per amplifier channel; two channels per physical package |
| OPA1641 | Texas Instruments, SBOM627 | OPA164x | One channel per physical OPA1641 package |
| LM741 | [Texas Instruments, SNOM211B](https://www.ti.com/lit/zip/snom211) | LM741 | Original National Semiconductor macromodel; one channel per package; no offset-null pins |
| 2N3904 | onsemi, 2N3904.LIB | Q2n3904 | C/B/E pin order |
| 2N3906 | onsemi, 2N3906.LIB | q2n3906 | C/B/E pin order |
| 1N4148 | ngspice basic_models collection | D1N4148/TEMP | A/K pin order; source header documents model limitations |

The simulator runs with an isolated ngspice startup file and PSpice compatibility settings. It does not modify the user's ngspice configuration. Checksums establish model identity, not electrical fidelity. Nominal simulation and the optional declared tolerance cases do not establish manufacturing yield or full device corner coverage.

The LM741 archive member is `lm741.lib`, with subcircuit terminals `1 2 99 50 28`: noninverting input, inverting input, positive supply, negative supply, output. These are model terminal identifiers, not the eight-pin package numbers. The downloaded model remains unchanged. It is an LM741 model, not an independently verified model for every device sold as a 741 or for TI's UA741.

The LM741 catalog uses a conservative 20–44 V total supply span from the ±10 to ±22 V recommended dual-supply range in [TI LM741 datasheet Rev. D, section 6.3, page 4](https://www.ti.com/lit/ds/symlink/lm741.pdf), and a 0–70 °C range shared by the LM741/LM741C specifications. TI's product summary lists a lower minimum supply than that table; the harness uses the table. A span check does not establish valid common-mode voltage, symmetric rails, output headroom, safe dissipation, or model accuracy throughout this envelope.

The original LM741 macromodel has supply-referenced stages and a common-mode stage, but does not identify a separately characterized PSRR stage or a validated rail-noise frequency range. Rail-to-output coupling is therefore an unvalidated model prediction. A small or zero simulated response does not establish physical immunity to supply noise. Datasheet section 6.5's supply-voltage rejection specification uses a DC supply change and cannot validate an AC rail-noise spectrum by itself. Signal gain and rail attenuation can be measured in the harness; claims of datasheet-accurate PSRR, stochastic device noise, radio-frequency rail behavior, or measured laboratory equivalence require independent validation.
