"""Saved samples produce honest, bounded, per-evaluation output plots."""
import io
import json
import shutil
import zipfile
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import pytest

from spiceharness.saved_outputs import _decimate, render_saved_outputs


def _folder(tmp_path):
    path = tmp_path / ".spiceharness" / "experiments" / "run" / "evaluation"
    path.mkdir(parents=True)
    return path


def _spec():
    return {"probes": [{"name": "load_response", "positive": "sense_plus", "negative": "sense_minus"}],
            "transient": {"start_s": 0.2, "stop_s": 0.8},
            "sources": [{"ref": "drive", "kind": "V"}],
            "measurements": [{"operator": "transient_mean", "probe": "load_response"}]}


def test_arbitrary_differential_probe_and_complex_ac_output(tmp_path):
    path = _folder(tmp_path)
    time = np.linspace(0, 1, 1201)
    positive = 2+np.sin(2*np.pi*time)
    negative = np.ones(len(time))
    np.savez_compressed(path / "2_tran.curves.npz", names=np.array(["time", "v(sense_plus)", "v(sense_minus)"]),
                        data=np.column_stack([time, positive, negative]).astype(complex))
    frequency = np.geomspace(10, 10000, 1000)
    np.savez_compressed(path / "1_ac_drive.curves.npz",
                        names=np.array(["frequency", "v(sense_plus)", "v(sense_minus)"]),
                        data=np.column_stack([frequency, 2/(1+1j*frequency/100), np.zeros(len(frequency))]))
    plots = render_saved_outputs({"artifact_dir": str(path)}, workspace=tmp_path, spec=_spec())
    assert len(plots) == 2
    assert plots[0]["title"] == "Transient output · load_response"
    assert plots[0]["sample_count"] == 721
    assert plots[1]["source"] == "drive"
    assert "V(sense_plus) − V(sense_minus)" in plots[0]["note"]
    assert "Frequency (Hz), log scale" in plots[1]["svg"]
    assert "Magnitude (V/V)" in plots[1]["svg"]
    assert "Voltage (V)" in plots[0]["svg"]
    for plot in plots:
        ElementTree.fromstring(plot["svg"])
        assert "<script" not in plot["svg"]
        assert "No new simulation or score" in plot["note"]


def test_decimation_preserves_narrow_positive_and_negative_extrema():
    time = np.arange(20_000)
    signal = np.zeros(len(time))
    signal[3456], signal[3457] = 1000, -2000
    x, y = _decimate(time, signal)
    assert len(x) <= 500
    assert x[0] == time[0] and x[-1] == time[-1]
    assert np.max(y) == 1000
    assert np.min(y) == -2000
    assert np.all(np.diff(x) > 0)


def test_legacy_ascii_raw_and_output_node_fallback(tmp_path):
    path = _folder(tmp_path)
    (path / "tran.raw").write_text(
        "Title: saved ngspice evidence\nNo. Variables: 3\nNo. Points: 3\nVariables:\n"
        "0 time time\n1 v(in) voltage\n2 v(out) voltage\nValues:\n"
        "0 0 1 0\n1 0.001 1 0.5\n2 0.002 1 0.75\n")
    plots = render_saved_outputs({"artifact_dir": str(path)}, workspace=tmp_path,
                                 circuit={"components": [{"nodes": ["in", "out"]}]})
    assert [plot["probe"] for plot in plots] == ["out", "in"]
    assert plots[0]["sample_count"] == 3


def test_untrusted_paths_symlink_files_and_directories_are_not_read(tmp_path):
    path = _folder(tmp_path)
    source = tmp_path / "outside.npz"
    np.savez(source, names=np.array(["time", "v(out)"]), data=np.array([[0, 0], [1, 1]]))
    (path / "tran.curves.npz").symlink_to(source)
    assert render_saved_outputs({"artifact_dir": str(path)}, workspace=tmp_path) == []
    assert render_saved_outputs({"artifact_dir": str(tmp_path)}, workspace=tmp_path) == []
    assert render_saved_outputs({"artifact_dir": str(path)}) == []
    link = path.parent / "linked"
    link.symlink_to(path, target_is_directory=True)
    assert render_saved_outputs({"artifact_dir": str(link)}, workspace=tmp_path) == []


def test_npz_headers_reject_allocation_bombs_and_pickle_before_load(tmp_path, monkeypatch):
    path = _folder(tmp_path)
    headers = {}
    for name, shape, dtype in (("names.npy", (2,), "<U10"), ("data.npy", (10**12, 2), "<f8")):
        stream = io.BytesIO()
        np.lib.format.write_array_header_1_0(stream, {"descr": dtype, "fortran_order": False, "shape": shape})
        headers[name] = stream.getvalue()
    with zipfile.ZipFile(path / "tran.curves.npz", "w") as archive:
        for name, content in headers.items():
            archive.writestr(name, content)

    def forbidden_load(*args, **kwargs):
        raise AssertionError("Unsafe dimensions must be rejected before np.load")

    monkeypatch.setattr(np, "load", forbidden_load)
    assert render_saved_outputs({"artifact_dir": str(path)}, workspace=tmp_path) == []


def test_bad_and_nonfinite_samples_do_not_become_synthetic_plots(tmp_path):
    path = _folder(tmp_path)
    np.savez(path / "tran.curves.npz", names=np.array(["time", "v(out)"]),
             data=np.array([[0, 0], [1, np.nan]]))
    (path / "ac.raw").write_text("No. Variables: 10000000000\nNo. Points: 10000000000000\n")
    assert render_saved_outputs({"artifact_dir": str(path)}, workspace=tmp_path) == []


def test_plot_text_escapes_untrusted_saved_probe_names(tmp_path):
    path = _folder(tmp_path)
    np.savez(path / "tran.curves.npz", names=np.array(["time", "v(sense_plus)", "v(sense_minus)"]),
             data=np.array([[0.2, 2, 1], [0.8, 3, 1]]))
    spec = _spec()
    spec["probes"][0]["name"] = '</text><script>alert("x")</script>'
    plots = render_saved_outputs({"artifact_dir": str(path)}, workspace=tmp_path, spec=spec)
    assert len(plots) == 1
    assert "<script>" not in plots[0]["svg"]
    assert "&lt;script&gt;" in plots[0]["svg"]
    ElementTree.fromstring(plots[0]["svg"])


@pytest.mark.skipif(not shutil.which("ngspice"), reason="real ngspice required")
def test_real_saved_rc_samples_render_without_rerunning_evaluator(tmp_path):
    from spiceharness.experiments import ExperimentEvaluator, ExperimentSpec, GeneralCircuitIR

    examples = Path(__file__).resolve().parents[1] / "examples" / "experiments" / "loaded-rc"
    spec = ExperimentSpec.model_validate(json.loads((examples / "spec.json").read_text()))
    circuit = GeneralCircuitIR.model_validate(json.loads((examples / "reference.json").read_text()))
    result = ExperimentEvaluator(artifact_root=tmp_path / ".spiceharness" / "samples").evaluate(circuit, spec)
    assert result.status == "ok", result.errors
    plots = render_saved_outputs(result.model_dump(mode="json"), workspace=tmp_path,
                                 spec=spec.model_dump(mode="json"), circuit=circuit.model_dump(mode="json"))
    assert plots
    assert plots[0]["analysis"] == "ac"
    assert plots[0]["source"] == "signal"
    assert plots[0]["sample_count"] > 500


def test_plot_display_limit_keeps_primary_output_from_each_analysis(tmp_path):
    path = _folder(tmp_path)
    probes = [{"name": f"output_{i}", "positive": f"n{i}", "negative": "0"} for i in range(32)]
    names = np.array(["frequency", *(f"v(n{i})" for i in range(32))])
    data = np.array([[10, *range(32)], [100, *range(1, 33)]], dtype=complex)
    for index in range(3):
        np.savez(path / f"{index}_ac_drive{index}.curves.npz", names=names, data=data)
    plots = render_saved_outputs({"artifact_dir": str(path)}, workspace=tmp_path, spec={"probes": probes})
    assert len(plots) == 64
    assert [plot["source"] for plot in plots[:3]] == ["drive0", "drive1", "drive2"]
    assert all(plot["probe"] == "output_0" for plot in plots[:3])
    assert "primary output from each analysis is retained" in plots[-1]["note"]
