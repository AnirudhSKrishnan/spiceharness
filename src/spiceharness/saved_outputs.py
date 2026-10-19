"""Bounded, offline SVG plots of existing simulator samples, never new analyses."""
from __future__ import annotations

import math
import re
import zipfile
from html import escape
from pathlib import Path

import numpy as np

from .simulator import _read_raw

_MAX_BYTES = 64 * 1024 * 1024
_MAX_CELLS = 4_000_000
_MAX_RAW_BYTES = 16 * 1024 * 1024
_MAX_RAW_CELLS = 1_000_000
_MAX_SAMPLES = 250_000
_MAX_VECTORS = 256
_MAX_PLOTS = 64


def _directory(simulation: dict, workspace: Path | None) -> Path | None:
    if workspace is None or not isinstance(simulation.get("artifact_dir"), str):
        return None
    workspace = Path(workspace).expanduser().resolve()
    root = workspace / ".spiceharness"
    path = Path(simulation["artifact_dir"]).expanduser()
    if not path.is_absolute():
        path = workspace / path
    resolved = path.resolve()
    if not resolved.is_relative_to(root) or not resolved.is_dir():
        return None
    # Reject symlinked directories as well as symlinked files, including links
    # that happen to lead back into the artifact tree.
    current = path
    while current != workspace and current != current.parent:
        if current.is_symlink():
            return None
        current = current.parent
    return resolved


def _header(stream) -> tuple:
    version = np.lib.format.read_magic(stream)
    if version == (1, 0):
        return np.lib.format.read_array_header_1_0(stream)
    if version == (2, 0):
        return np.lib.format.read_array_header_2_0(stream)
    raise ValueError("Unsupported stored array version")


def _samples(path: Path) -> tuple[list[str], np.ndarray]:
    if path.stat().st_size > _MAX_BYTES:
        raise ValueError("Stored curve file exceeds display size limit")
    if path.suffix == ".npz":
        with zipfile.ZipFile(path) as archive:
            entries = archive.infolist()
            if {entry.filename for entry in entries} != {"names.npy", "data.npy"} or len(entries) != 2:
                raise ValueError("Unexpected curve archive entries")
            if sum(entry.file_size for entry in entries) > _MAX_BYTES:
                raise ValueError("Stored curve archive exceeds uncompressed display limit")
            for entry in entries:
                with archive.open(entry) as stream:
                    shape, _, dtype = _header(stream)
                if dtype.hasobject or dtype.itemsize > 1024:
                    raise ValueError("Unsafe stored curve dtype")
                if entry.filename == "names.npy":
                    if len(shape) != 1 or not 0 < shape[0] <= _MAX_VECTORS or dtype.kind not in "US":
                        raise ValueError("Invalid stored vector names")
                elif (len(shape) != 2 or not 0 < shape[0] <= _MAX_SAMPLES
                      or not 0 < shape[1] <= _MAX_VECTORS or math.prod(shape) > _MAX_CELLS
                      or dtype.kind not in "fcui"):
                    raise ValueError("Stored curve dimensions exceed display limits")
        with np.load(path, allow_pickle=False) as values:
            names = [str(value).lower() for value in values["names"]]
            data = values["data"]
    else:
        if path.stat().st_size > _MAX_RAW_BYTES:
            raise ValueError("Stored ASCII raw exceeds display size limit")
        # Inspect the dimensions before the evaluator's ASCII parser allocates.
        with path.open(encoding="utf-8", errors="strict") as handle:
            header = handle.read(65536).split("Values:", 1)[0]
        count = re.search(r"No\. Variables:\s*(\d+)", header)
        points = re.search(r"No\. Points:\s*(\d+)", header)
        if not count or not points:
            raise ValueError("Missing ASCII raw dimensions")
        vectors, samples = int(count[1]), int(points[1])
        if not (0 < vectors <= _MAX_VECTORS and 0 < samples <= _MAX_SAMPLES
                and vectors * samples <= _MAX_RAW_CELLS):
            raise ValueError("Stored raw dimensions exceed display limits")
        names, data = _read_raw(path)
    if data.ndim != 2 or data.shape[1] != len(names) or not np.isfinite(data).all():
        raise ValueError("Invalid or nonfinite stored samples")
    return names, data


def _decimate(x: np.ndarray, y: np.ndarray, limit: int = 500) -> tuple[np.ndarray, np.ndarray]:
    """Keep ordered min/max samples in each bin, including both endpoints."""
    if len(x) <= limit:
        return x, y
    chosen = {0, len(x)-1}
    bins = np.array_split(np.arange(1, len(x)-1), (limit-2)//2)
    for bucket in bins:
        if len(bucket):
            chosen.add(int(bucket[np.argmin(y[bucket])]))
            chosen.add(int(bucket[np.argmax(y[bucket])]))
    indices = sorted(chosen)
    return x[indices], y[indices]


def _svg(x: np.ndarray, y: np.ndarray, title: str, *, ac: bool, unit: str) -> str:
    x, y = _decimate(x, y)
    axis = np.log10(x) if ac else x
    xmin, xmax = float(axis.min()), float(axis.max())
    ymin, ymax = float(y.min()), float(y.max())
    if xmax <= xmin:
        raise ValueError("A curve requires increasing independent samples")
    pad = (ymax-ymin)*0.08 if ymax > ymin else max(abs(ymin)*0.05, 1e-9)
    ymin, ymax = ymin-pad, ymax+pad
    left, top, width, height = 95, 52, 850, 285
    px = left + (axis-xmin)/(xmax-xmin)*width
    py = top + (ymax-y)/(ymax-ymin)*height
    line = " ".join(f"{'M' if i == 0 else 'L'}{a:.2f},{b:.2f}" for i, (a, b) in enumerate(zip(px, py)))
    title = escape(title, quote=True)
    output = [(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 415" '
              f'width="1000" height="415" role="img" aria-label="{title}">'),
              '<rect width="1000" height="415" fill="#fff"/>',
              f'<text x="24" y="28" fill="#111" font-family="sans-serif" font-size="16">{title}</text>']
    for i in range(6):
        tx = left + width*i/5
        value = 10**(xmin+(xmax-xmin)*i/5) if ac else xmin+(xmax-xmin)*i/5
        output += [f'<path d="M{tx:.2f},{top} v{height}" stroke="#ddd"/>',
                   (f'<text x="{tx:.2f}" y="363" text-anchor="middle" fill="#666" '
                   f'font-family="monospace" font-size="12">{value:.4g}</text>')]
    for i in range(5):
        ty = top + height*i/4
        value = ymax-(ymax-ymin)*i/4
        output += [f'<path d="M{left},{ty:.2f} h{width}" stroke="#ddd"/>',
                   (f'<text x="84" y="{ty+4:.2f}" text-anchor="end" fill="#666" '
                   f'font-family="monospace" font-size="12">{value:.4g}</text>')]
    output += [f'<path d="{line}" fill="none" stroke="#111" stroke-width="1.5"/>',
               ('<text x="520" y="396" text-anchor="middle" fill="#111" '
               f'font-family="sans-serif" font-size="13">{"Frequency (Hz), log scale" if ac else "Time (s)"}</text>'),
               ('<text x="20" y="200" transform="rotate(-90 20 200)" text-anchor="middle" fill="#111" '
               f'font-family="sans-serif" font-size="13">{escape(unit)}</text>'), '</svg>']
    return "".join(output)


def _probes(spec: dict, circuit: dict, names: list[str]) -> list[dict]:
    probes = [probe for probe in spec.get("probes", []) if isinstance(probe, dict)]
    if not probes:
        # Legacy Circuit IR uses the canonical input/output node names. If a
        # historic circuit exposes others, show its saved voltage vectors too.
        nodes = {node.lower() for part in circuit.get("components", []) for node in part.get("nodes", [])}
        available = [name[2:-1] for name in names if name.startswith("v(") and name.endswith(")")]
        wanted = [node for node in ("out", "output", "in", "input") if node in available]
        wanted += [node for node in available if node in nodes and node not in wanted]
        probes = [{"name": node, "positive": node, "negative": "0"} for node in wanted[:32]]
    # Put the specification's output measurement first, without assuming that it
    # is named 'out'. This controls presentation only, never scoring.
    output = [m.get("probe") for m in spec.get("measurements", [])
              if str(m.get("operator", "")).startswith("transient_")]
    return sorted(probes, key=lambda p: (p.get("name") not in output,
                                        not any(word in str(p.get("name", "")).lower() for word in ("out", "output"))))


def render_saved_outputs(simulation: dict, *, workspace: Path | None = None,
                         spec: dict | None = None, circuit: dict | None = None) -> list[dict]:
    """Return safe plot SVGs of each candidate's existing AC/transient evidence.

    Arrays and ASCII raw files are size/dimension bounded, loaded without pickle,
    and restricted to regular, non-symlink files in the workspace artifact tree.
    Decimation retains local extrema; the original samples remain unchanged.
    """
    directory = _directory(simulation, workspace)
    if directory is None:
        return []
    spec, circuit = spec or {}, circuit or {}
    files = sorted(directory.glob("*.curves.npz"))
    covered = {path.name.removesuffix(".curves.npz") for path in files}
    files += sorted(path for path in directory.glob("*.raw") if path.stem not in covered)
    # Prefer transient output as the initial plot, then all stored AC analyses.
    files.sort(key=lambda p: ("tran" not in p.name.lower(), p.name))
    groups = []
    for path in files[:18]:
        result = []
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(directory):
            continue
        try:
            names, values = _samples(path)
            if not names or names[0] not in {"time", "frequency"} or len(values) < 2:
                continue
            ac = names[0] == "frequency"
            x = values[:, 0].real
            if not np.isfinite(x).all() or np.any(np.diff(x) < 0) or (ac and np.any(x <= 0)):
                continue
            source_match = re.search(r"(?:^|_)ac_(.+?)(?:\.curves\.npz|\.raw)$", path.name)
            source = source_match[1] if source_match else "input"
            source_spec = next((s for s in spec.get("sources", []) if s.get("ref") == source), {})
            unit = ("Magnitude (V/A)" if source_spec.get("kind") == "I" else "Magnitude (V/V)") if ac else "Voltage (V)"
            for probe in _probes(spec, circuit, names):
                positive, negative = str(probe.get("positive", "")).lower(), str(probe.get("negative", "0")).lower()

                def voltage(node, values=values, names=names):
                    return np.zeros(len(values), dtype=complex) if node == "0" else values[:, names.index(f"v({node})")]

                try:
                    vector = voltage(positive)-voltage(negative)
                except ValueError:
                    continue
                y = np.abs(vector) if ac else vector.real
                mask = np.ones(len(x), dtype=bool)
                if not ac and spec.get("transient"):
                    window = spec["transient"]
                    mask = (x >= window.get("start_s", float(x[0]))) & (x <= window.get("stop_s", float(x[-1])))
                sx, sy = x[mask], y[mask]
                if len(sx) < 2 or not np.isfinite(sy).all():
                    continue
                label = str(probe.get("name") or positive)
                title = f"AC · source {source} · {label}" if ac else f"Transient output · {label}"
                note = (f"Saved {path.name}; probe V({positive}) − V({negative}). "
                        f"{len(sx)} stored samples in the plotted window; at most 500 displayed points "
                        "retain each display bin's minimum and maximum. No new simulation or score. "
                        "AC traces are sampled nominal magnitudes, not exhaustive worst-case analysis." if ac else
                        f"Saved {path.name}; probe V({positive}) − V({negative}). "
                        f"{len(sx)} stored samples in the saved measurement window; at most 500 displayed points "
                        "retain each display bin's minimum and maximum. No new simulation or score.")
                result.append({"title": title, "svg": _svg(sx, sy, title, ac=ac, unit=unit), "note": note,
                               "analysis": "ac" if ac else "transient", "probe": label,
                               "source": source if ac else None, "sample_count": len(sx)})
            if result:
                groups.append(result)
        except (OSError, ValueError, TypeError, KeyError, IndexError, AttributeError,
                OverflowError, EOFError, zipfile.BadZipFile):
            # A missing or malformed curve is not an evaluation failure and must
            # never turn into a synthesized curve or a fabricated pass.
            continue
    # Interleave probes so every supported stored analysis keeps its primary
    # output curve even when the bounded display cannot include every probe.
    result = []
    for index in range(max((len(group) for group in groups), default=0)):
        for group in groups:
            if index < len(group):
                result.append(group[index])
                if len(result) >= _MAX_PLOTS:
                    result[-1]["note"] += " Display limited to 64 plots; primary output from each analysis is retained."
                    return result
    return result
