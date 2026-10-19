# Re-scoring an edited KiCad schematic

`spiceharness export RUN` writes an editable `candidate.kicad_sch` for every saved candidate. After you edit one in
KiCad, the importer reads it back, rebuilds the Circuit IR from the **drawn connectivity**, and re-scores it as a
**new candidate in a new run** under the **same frozen specification**. The edit never inherits the original score,
and the original run is never modified. No agents are called; scores come only from the evaluator and ngspice.

## Workflow

```sh
spice export RUN                                  # or: uv run spiceharness export RUN
# open .spiceharness/exports/RUN/<candidate>/candidate.kicad_sch in KiCad, edit, save
spice experiment rescore-kicad RUN .spiceharness/exports/RUN/<candidate>/candidate.kicad_sch
```

`RUN` may be a unique prefix. Options: `-w WORKSPACE`, `--json`, `--name NAME`. The command:

1. copies the original run's frozen spec (the new run's fixture fingerprint must equal the original's);
2. imports the schematic and refuses to start anything when the fixture changed or the circuit is invalid;
3. starts a new run, records the original run ID, schematic path and SHA-256 (event `kicad_rescore_import`,
   `kicad_import.json`, and a copy `imported.kicad_sch` in the new run folder);
4. evaluates the circuit and freshly verifies it (`finalize`);
5. prints valid / feasible / verified, every frozen requirement as PASS or FAIL, the original winner's metrics and
   `rank_key` beside the edit's, and where the edit would have placed on the original leaderboard.

Exit status is 1 when a scored run is not verified and feasible. An analysis-only run (no acceptance requirements)
prints measurements only: "Not scored (analysis only, no acceptance requirements)". Template `design` runs are
refused ("only general experiment runs are supported").

Through MCP, `import_kicad_schematic(run_id, path)` is **read-only**: it returns the rebuilt IR, the change list,
warnings, `fixture_ok` and any evaluator validation error. Submit the IR with `evaluate_experiment` in a new run
started from the same spec. The path must be inside the workspace or in an export folder (a folder with this run's
`manifest.json`), with no symlink on the way.

## Supported edits

* Change R, C, L values (Value field and/or `Sim.Params`), including intrinsic ESR/DCR resistors.
* Add, delete or swap R/C/L parts: the exporter's `SpiceHarness:*` symbols or KiCad stock `Device:R`, `Device:C`,
  `Device:L` and their `_Small` (and `R_US`, `L_US`) variants. New parts take their KiCad Reference as IR `ref`
  (annotate first; references must be simple identifiers and unique).
* Change an op-amp, BJT or diode model to another model **in the trusted catalog** (Value or `Sim.Name`).
* Rewire with wires, junctions, local labels, global labels, `GND` power symbols. Rotate and mirror symbols freely.
* Add or remove exporter-style ideal connections (a pin-less wire carrying two different labels).

Connectivity follows KiCad 10.0.7 (verified against `kicad-cli sch export netlist`): items sharing a coordinate
(wire ends, pins, labels, junctions) connect; a wire end or pin that only touches the **middle** of another wire,
and crossing wires, connect **only through a junction** (the importer warns about near misses); a label sitting on
a wire connects to it; labels with the same text connect (local, global and power symbols share one
case-sensitive namespace; a power symbol is named by its Value). `GND`, `gnd` and `0` are node `0`. Labels that
differ only in case are an error (KiCad keeps them apart, SpiceHarness nodes are case-insensitive). Unlabeled
nets get generated names `net1`, `net2`, ... A wire that joins two differently labeled nets through component pins
makes them one node (with a warning); a pin-less wire carrying two labels becomes an ideal connection.

IR details not stored in the schematic (`package`, `role`, component and connection order, circuit name) come from
the frozen spec's baseline for protected parts, otherwise from the `circuit.json` beside the schematic, otherwise
defaults. Unedited export imports back to the identical IR.

## Value rule

Values come from `Sim.Params` (`r=`, `c=`, `l=`) and/or the Value field, SPICE/SI style: `T G Meg k m u µ n p f`,
RKM forms (`4k7`, `2R2`), trailing units (`100µF`, `2.2kΩ`, `10nH`).

**`M` is mega and `m` is milli, as in KiCad 10.** `Meg`, `MEG` and `meg` are mega. This differs from plain SPICE,
where `M` is milli; KiCad 10.0.7 itself writes `1M` as `1Meg` into its SPICE deck (checked with `kicad-cli`), so
the importer agrees with what KiCad simulates. Every bare `M` is accepted with a warning that names the reading;
lowercase `m` on a resistor warns too (milli-ohms). `10F` is rejected as ambiguous (femto vs farad); write `10f` or `10`.

If Value and `Sim.Params` disagree, the importer compares each with the original `circuit.json`: if only one was
edited it uses the edited one (with a warning when only the Value field was edited); if it cannot tell, it stops with
an error rather than score a stale number. KiCad's own simulator uses `Sim.Params`, so edit both.

## What is rejected, and why

| Rejected | Why |
| --- | --- |
| Edited or deleted fixture source/load; fixture on another net; extra fixture symbol | The bench is frozen by the spec. `fixture_ok` is false with a precise message (`fixture RLOAD: Value gives 5 kΩ but the frozen load is 10 kΩ`). |
| New V/I source; DNP, excluded-from-simulation or `Sim.Enable=0` parts | Sources are fixtures. Delete a part instead of disabling it, so nothing is silently dropped. |
| Unknown `lib_id`, polarized caps, multi-unit symbols, sheets, hierarchical labels, buses, net-class flags, unknown items | Not representable in the Circuit IR. |
| Model not in the catalog, or a custom `Sim.Library` | No ideal or arbitrary substitute models; only trusted catalog models. |
| Behavioural passives (`Sim.Name`), extra `Sim.Params` (e.g. `tc1`) | Raw SPICE and arbitrary parameters are never accepted. SPICE text directives are ignored with a warning. |
| Intrinsic-loss resistor no longer in series with its parent on a private net | The IR can only fold it into `series_resistance_ohm`. Deleting it means zero loss. |
| No-connect flag on a used pin; unconnected modeled pins; unannotated or duplicate references | Every modeled terminal must be connected; copies of exported symbols keep their `IR.Reference`, so clear the `IR.*` properties of a copy. |
| Shorts across a source, floating islands, protected host parts changed | Rejected by the existing `ExperimentEvaluator.validate` before any simulation. |

Nominal sampled results are not a worst-case proof; see the experiment guide for limits.
