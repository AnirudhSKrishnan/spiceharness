"""Offline, read-only browser snapshots of saved deterministic evaluations.

This module never evaluates circuits or recalculates scores. Only files already
saved beneath the workspace's .spiceharness artifact tree may be linked/read.
"""
from __future__ import annotations

import json
import math
import re
import shlex
from pathlib import Path
from typing import Any

from .circuit_svg import render_circuit_svg
from .saved_outputs import render_saved_outputs
from .storage import utcnow
from .workspace import command_hint


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _json(value: Any) -> str:
    # The payload is in a script raw-text element: HTML escaping alone is insufficient.
    return (json.dumps(value, ensure_ascii=True, allow_nan=False)
            .replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e"))


def _clean(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    return value


def _bound_text(low: Any, high: Any) -> str:
    if low is None and high is None:
        return "No requirement"
    if low is None:
        return f"≤ {high:.8g}"
    if high is None:
        return f"≥ {low:.8g}"
    return f"{low:.8g} ≤ value ≤ {high:.8g}"


def _metric_rows(spec: dict, simulation: dict, score: dict, general: bool) -> list[dict]:
    metrics = simulation.get("metrics") or score.get("metrics") or {}
    violations = score.get("violations") or {}
    rows = []

    def add(name, value, bound, key=None, scope="nominal"):
        violation = violations.get(key) if key is not None else None
        state = ("pass" if violation == 0 else "fail") if _number(violation) and _number(value) else (
            "analysis-only" if key is None else "unknown")
        rows.append({"metric": name, "scope": scope, "value": value, "bound": bound,
                     "violation": violation, "state": state, "key": key})

    if general:
        constrained = set()
        for index, requirement in enumerate(spec.get("requirements") or []):
            name = requirement.get("metric", "unknown")
            constrained.add(name)
            add(name, metrics.get(name), _bound_text(requirement.get("minimum"), requirement.get("maximum")),
                f"{name}:{index}")
        for name, value in metrics.items():
            if name not in constrained:
                add(name, value, "No requirement")
        return rows

    # Mirror only supported legacy constraint descriptions. Pass/fail and normalized
    # violations still come exclusively from the recorded evaluator score.
    desired = abs(spec.get("gain") if spec.get("gain") is not None else 1)
    tolerance = spec.get("gain_tolerance", 0.05)
    bounds = {"gain": ("gain", _bound_text(desired * (1-tolerance), desired * (1+tolerance))),
              "thd": ("thd", _bound_text(None, spec.get("max_thd", 0.03))),
              "gain_compression": ("gain_compression", "≤ 0.05")}
    if spec.get("cutoff_hz") is not None:
        cutoff, tolerance = spec["cutoff_hz"], spec.get("cutoff_tolerance", 0.1)
        bounds["cutoff_hz"] = ("cutoff_hz", _bound_text(cutoff*(1-tolerance), cutoff*(1+tolerance)))
    for field, metric in (("max_stopband_gain", "stopband_gain"),
                          ("max_supply_current_a", "supply_current_a"),
                          ("max_output_offset_v", "output_offset_v")):
        if spec.get(field) is not None:
            bounds[metric] = (metric, ("absolute value " if metric == "output_offset_v" else "")
                              + _bound_text(None, spec[field]))
    if spec.get("min_output_swing_v") is not None:
        bounds["minimum_output_swing_v"] = ("output_swing_v", _bound_text(spec["min_output_swing_v"], None))
    if spec.get("gain") is not None and "pass" not in spec.get("topology", ""):
        expected = 180 if spec["gain"] < 0 else 0
        bounds["gain_polarity"] = ("phase_deg", f"circular distance from {expected}° ≤ 45°")
    scopes = {"nominal": metrics}
    for label, entry in (simulation.get("analyses", {}).get("corners") or {}).items():
        scopes[label] = entry.get("metrics") or {}
    used = set()
    for scope, measured in scopes.items():
        constrained = set()
        for constraint, (metric, description) in bounds.items():
            key = f"{scope}.{constraint}"
            constrained.add(metric)
            used.add(key)
            add(metric, measured.get(metric), description, key, scope)
        for metric, value in measured.items():
            if metric not in constrained:
                add(metric, value, "No requirement", scope=scope)
    if spec.get("max_parts") is not None:
        used.add("package_count")
        add("package_count", score.get("package_count"), _bound_text(None, spec["max_parts"]), "package_count")
    for key in violations.keys() - used:
        add(key, None, "Recorded evaluator constraint; bound unavailable", key)
    return rows


def _artifacts(simulation: dict, workspace: Path | None) -> list[dict]:
    if workspace is None:
        return []
    root = (workspace / ".spiceharness").resolve()
    directory = simulation.get("artifact_dir")
    if not isinstance(directory, str) or not directory:
        return []
    path = Path(directory).expanduser()
    path = (workspace / path).resolve() if not path.is_absolute() else path.resolve()
    if not path.is_relative_to(root) or not path.is_dir():
        return []
    # Original evaluator artifact list when present; legacy evaluators save decks
    # in the artifact directory without an explicit file list.
    names = simulation.get("artifacts")
    paths = [path / name for name in names if isinstance(name, str)] if isinstance(names, list) else list(path.iterdir())
    result = []
    allowed = {".cir", ".log", ".txt", ".json", ".csv", ".raw", ".npz", ".md"}
    for entry in sorted(paths, key=str):
        resolved = entry.resolve()
        if not resolved.is_relative_to(path) or not resolved.is_relative_to(root):
            continue
        if not resolved.is_file() or resolved.suffix.lower() not in allowed:
            continue
        item = {"name": str(entry.relative_to(path)), "path": str(resolved), "href": resolved.as_uri()}
        if resolved.suffix.lower() == ".cir":
            item["deck"] = resolved.read_text(encoding="utf-8", errors="replace")[:200_000]
        result.append(item)
    return result



def _export_file(root: Path, name: Any) -> Path | None:
    if not isinstance(name, str) or not name:
        return None
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        return None
    path = root / relative
    try:
        if not path.resolve().is_relative_to(root.resolve()):
            return None
        if any(part.is_symlink() for part in (path, *path.parents)):
            return None
        return path if path.is_file() else None
    except OSError:
        return None


def _kicad_exports(run: dict, workspace: Path | None) -> dict:
    run_id = run.get("run_id")
    if workspace is None or not isinstance(run_id, str) or not re.fullmatch(r"[0-9a-f]{32}", run_id):
        return {}
    root = workspace / ".spiceharness" / "exports"
    try:
        manifests = []
        for folder in root.iterdir():
            if re.fullmatch(re.escape(run_id) + r"(?:-[0-9]+)?", folder.name):
                manifest = _export_file(root, folder.name + "/manifest.json")
                if manifest is not None and manifest.stat().st_size <= 2 * 1024 * 1024:
                    manifests.append(manifest)
        for manifest in sorted(manifests, key=lambda path: path.stat().st_mtime_ns, reverse=True):
            try:
                value = json.loads(manifest.read_text(encoding="utf-8"))
            except (OSError, ValueError, UnicodeError):
                continue
            if not isinstance(value, dict) or value.get("run_id") != run_id or not isinstance(value.get("candidates"), list):
                continue
            result = {}
            for entry in value["candidates"]:
                if not isinstance(entry, dict) or not isinstance(entry.get("candidate_id"), str):
                    continue
                schematic = _export_file(manifest.parent, entry.get("schematic"))
                files = []
                if schematic is not None and schematic.suffix == ".kicad_sch":
                    project = _export_file(manifest.parent, str(schematic.relative_to(manifest.parent).with_suffix(".kicad_pro")))
                    for label, path in (("Open project in KiCad", project), ("Open schematic", schematic)):
                        if path is not None:
                            files.append({"name": label, "path": str(path), "href": path.as_uri()})
                result[entry["candidate_id"]] = {"files": files,
                    "native_runnable": entry.get("native_runnable") is True,
                    "status": str(entry.get("status", "unknown")),
                    "missing": entry.get("missing") if isinstance(entry.get("missing"), list) else []}
            return result
    except OSError:
        pass
    return {}


def _viewer_data(run: dict, workspace: Path | None) -> dict:
    spec = run.get("spec") or {}
    result = run.get("result") or {}
    general = run.get("kind") == "experiment" or "requirements" in spec
    candidates = run.get("candidates") or []
    search = [row for row in candidates if row.get("purpose", "search") != "verification"]
    by_id = {row.get("candidate_id"): row for row in search}
    leaderboard = [by_id[row.get("candidate_id")] for row in (run.get("leaderboard") or [])
                   if row.get("candidate_id") in by_id and (row.get("score") or {}).get("rank_key") is not None]
    if not leaderboard:
        leaderboard = sorted((row for row in search if (row.get("score") or {}).get("rank_key") is not None),
                             key=lambda row: tuple(row["score"]["rank_key"]))
    if general and not spec.get("requirements"):
        leaderboard = []
    ranks = {row.get("candidate_id"): index + 1 for index, row in enumerate(leaderboard)}
    exports = _kicad_exports(run, workspace)
    rows = []
    for index, row in enumerate(candidates):
        circuit, simulation = row.get("circuit") or {}, row.get("simulation") or {}
        score = row.get("score") or simulation.get("score") or {}
        violation_values = [value for value in (score.get("violations") or {}).values() if _number(value)]
        purpose = row.get("purpose", "search")
        rows.append({"id": row.get("candidate_id", f"candidate-{index+1}"),
                     "name": circuit.get("name") or circuit.get("topology") or f"Candidate {index+1}",
                     "ordinal": row.get("ordinal", index+1), "rank": ranks.get(row.get("candidate_id")),
                     "purpose": purpose, "verified": bool(row.get("verified")) and purpose == "verification",
                     "status": simulation.get("status") or row.get("status") or "pending",
                     "score": score, "circuit": circuit,
                     "svg": render_circuit_svg(circuit, spec=spec),
                     "worst": score.get("max_violation", max(violation_values) if violation_values else None),
                     "metrics": _metric_rows(spec, simulation, score, general),
                     "errors": [*simulation.get("errors", []), *score.get("explanation", [])]
                               + ([row["error"]] if row.get("error") else []),
                     "artifacts": _artifacts(simulation, workspace),
                     "kicad": exports.get(row.get("candidate_id")),
                     "outputs": render_saved_outputs(simulation, workspace=workspace, spec=spec, circuit=circuit),
                     "fingerprints": {key: simulation[key] for key in
                                      ("circuit_fingerprint", "fixture_fingerprint", "spec_fingerprint",
                                       "model_checksums", "simulator_version") if key in simulation}})
    ids = {row["id"] for row in rows}
    verified_id = result.get("verification_candidate_id") if result.get("verified") else None
    default = verified_id if verified_id in ids else result.get("best_candidate_id")
    if default not in ids:
        default = leaderboard[0].get("candidate_id") if leaderboard else next(
            (row["id"] for row in reversed(rows) if row["score"]), rows[0]["id"] if rows else None)
    return _clean({"run_id": run.get("run_id", "unknown"), "status": run.get("status", "unknown"),
                   "mode": run.get("execution_mode") or ("offline" if run.get("offline") else "API"),
                   "offline": run.get("offline", False), "general": general,
                   "analysis_only": run.get("analysis_only", general and not spec.get("requirements")),
                   "verified": result.get("verified", False), "feasible": result.get("feasible"),
                   "created_at": run.get("created_at"), "updated_at": run.get("updated_at"),
                   "snapshot_at": utcnow(), "spec": spec, "candidates": rows, "default_id": default,
                   "best_id": result.get("best_candidate_id"), "verification_id": result.get("verification_candidate_id"),
                   "selection_policy": result.get("selection_policy"),
                   "export_command": command_hint("export " + shlex.quote(str(run.get("run_id", ""))), workspace),
                   "refresh_command": command_hint("view " + shlex.quote(str(run.get("run_id", ""))), workspace)})


def export_results_viewer(run: dict, destination: Path, *, workspace: Path | None = None) -> Path:
    """Write a self-contained HTML snapshot, without changing the saved run.

    ``workspace`` enables safe links and embedded generated decks from its
    ``.spiceharness`` artifact tree. Without it, the viewer performs no artifact I/O.
    """
    destination = Path(destination)
    data = _viewer_data(run, Path(workspace).expanduser().resolve() if workspace is not None else None)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(_HTML.replace("__PAYLOAD__", _json(data)), encoding="utf-8")
    return destination


_HTML = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data:; connect-src 'none'; object-src 'none'; base-uri 'none'; form-action 'none'">
<title>SpiceHarness · Results explorer</title><style>
:root{color-scheme:dark;--bg:#0a101a;--panel:#121c2b;--line:#29374b;--muted:#a0aec2;--ink:#edf4ff;--accent:#68c7ff;--pass:#73e2b0;--fail:#ff9c9c}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,sans-serif}header{padding:28px 32px 20px;border-bottom:1px solid var(--line);background:linear-gradient(110deg,#172b43,#111a29)}h1{font-size:27px;margin:3px 0 8px}h2{font-size:21px;margin:0 0 12px}h3{font-size:16px;margin:0 0 10px}p{margin:8px 0}.eyebrow{text-transform:uppercase;letter-spacing:.17em;color:var(--accent);font-size:12px;font-weight:700}.muted,small{color:var(--muted)}.badges{display:flex;flex-wrap:wrap;gap:9px;margin:12px 0}.badge{padding:3px 9px;border:1px solid var(--line);border-radius:7px;font-size:12px}.pass{color:var(--pass)}.fail{color:var(--fail)}.unknown,.analysis-only{color:#e4ca8b}.layout{display:grid;grid-template-columns:330px minmax(0,1fr);gap:24px;padding:24px 32px;align-items:start}aside{min-width:0}.candidate-list{display:grid;gap:9px}.candidate{background:var(--panel);color:var(--ink);border:1px solid var(--line);border-radius:10px;padding:14px;text-align:left;width:100%;cursor:pointer}.candidate:hover{border-color:var(--accent)}.candidate.selected{border:2px solid var(--accent);padding:13px;background:#182c42}.candidate strong{display:block;font-size:15px;overflow-wrap:anywhere}.candidate small{display:block;margin-top:5px}.candidate .line{display:flex;justify-content:space-between;gap:8px;margin-top:8px;font-size:12px}.search{width:100%;background:#0d1725;color:var(--ink);border:1px solid var(--line);border-radius:8px;padding:11px;margin:10px 0 14px}main{min-width:0}.panel{padding:22px;background:var(--panel);border:1px solid var(--line);border-radius:12px;margin-bottom:18px}.stats{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin:18px 0}.stat{padding:14px;background:#0d1725;border:1px solid var(--line);border-radius:9px}.stat small{display:block}.stat strong{font-size:21px}.tabs{display:flex;gap:8px;flex-wrap:wrap;margin:18px 0}.tab{border:1px solid var(--line);border-radius:7px;background:#0d1725;color:var(--muted);padding:10px 15px;cursor:pointer}.tab[aria-selected=true]{background:#254560;color:#fff;border-color:var(--accent)}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:13px}th{text-align:left;color:var(--muted);font-weight:500;background:#101928}th,td{padding:11px 12px;border-bottom:1px solid var(--line);vertical-align:top}td{overflow-wrap:anywhere}code,pre{font:12px/1.6 ui-monospace,SFMono-Regular,Consolas,monospace}pre{background:#09111d;border:1px solid var(--line);padding:16px;overflow:auto;border-radius:8px;max-height:650px;white-space:pre-wrap;overflow-wrap:anywhere}.schematic{background:#f3f7fb;border-radius:8px;overflow:auto;padding:12px;max-height:none}.schematic svg{display:block;min-width:650px;width:auto;max-width:none;height:auto}a{color:var(--accent)}summary{cursor:pointer;padding:10px 0}.notice{border-left:3px solid var(--accent);padding:9px 14px;background:#0e1c2c}.empty{padding:26px;text-align:center;color:var(--muted)}.split{display:grid;grid-template-columns:1fr 1fr;gap:16px}button:focus-visible,input:focus-visible,a:focus-visible{outline:3px solid #f2d18b;outline-offset:3px}footer{padding:0 32px 28px;color:var(--muted);font-size:12px}@media(max-width:1000px){.layout{grid-template-columns:270px minmax(0,1fr);padding:18px;gap:16px}.stats{grid-template-columns:repeat(2,1fr)}}@media(max-width:740px){header{padding:20px}.layout{display:block}.candidate-list{max-height:440px;overflow:auto;margin-bottom:20px}.split{grid-template-columns:1fr}.panel{padding:16px}.schematic svg{min-width:600px}h1{font-size:23px}}
</style></head><body><header><div class="eyebrow">SpiceHarness / saved results</div><h1>Candidate circuit explorer</h1><div id="run-info"></div><div id="run-badges" class="badges"></div><p class="muted" id="snapshot"></p></header>
<div class="layout"><aside><h2 id="candidate-count"></h2><label for="search" class="muted">Find a circuit or evaluation</label><input class="search" id="search" placeholder="Name, ID, search or verification"><div class="candidate-list" id="candidate-list"></div></aside><main id="detail"></main></div><footer>Read-only saved evidence · no network requests · no agents or simulator calls · SI units unless explicitly stated</footer>
<script type="application/json" id="run-data">__PAYLOAD__</script><script>
'use strict';
const data=JSON.parse(document.getElementById('run-data').textContent);let selected=data.default_id,activeTab='circuit';
const $=id=>document.getElementById(id);const el=(tag,text,cls)=>{const n=document.createElement(tag);if(text!==undefined)n.textContent=text;if(cls)n.className=cls;return n};
const fmt=v=>v===null||v===undefined?'unknown':typeof v==='number'?(Number.isInteger(v)?String(v):v.toPrecision(7)):typeof v==='object'?JSON.stringify(v):String(v);
const feasibility=c=>c.score.feasible===true?'feasible':c.score.feasible===false?'infeasible':data.analysis_only&&c.score.valid===true?'analysis-only':'unknown';
const badge=(parent,text,cls)=>parent.appendChild(el('span',text,'badge '+(cls||'')));
const table=(headers,rows)=>{const wrap=el('div',undefined,'scroll'),t=el('table'),head=el('thead'),tr=el('tr');headers.forEach(h=>tr.appendChild(el('th',h)));head.appendChild(tr);t.appendChild(head);const body=el('tbody');rows.forEach(row=>{const tr=el('tr');row.forEach(value=>{const td=el('td');if(value instanceof Node)td.appendChild(value);else td.textContent=fmt(value);tr.appendChild(td)});body.appendChild(tr)});t.appendChild(body);wrap.appendChild(t);return wrap};
function panel(title,parent){const box=el('section',undefined,'panel');if(title)box.appendChild(el('h3',title));parent.appendChild(box);return box}
function renderList(){const list=$('candidate-list');list.replaceChildren();const query=$('search').value.toLowerCase();data.candidates.filter(c=>(c.name+' '+c.id+' '+c.purpose).toLowerCase().includes(query)).forEach(c=>{const b=el('button',undefined,'candidate'+(c.id===selected?' selected':''));b.type='button';b.setAttribute('aria-pressed',c.id===selected);b.appendChild(el('strong',c.name));b.appendChild(el('small',(c.rank?'Rank '+c.rank+' · ':'')+(c.purpose==='verification'?'Fresh verification · same circuit':'Search evaluation '+c.ordinal)));const line=el('div',undefined,'line');line.appendChild(el('span',feasibility(c),feasibility(c)==='feasible'?'pass':feasibility(c)==='infeasible'?'fail':'unknown'));line.appendChild(el('span',fmt(c.score.package_count)+' packages'));b.appendChild(line);b.appendChild(el('small','Worst violation '+fmt(c.worst)+' · '+c.status));b.onclick=()=>{selected=c.id;renderList();renderDetail()};list.appendChild(b)});if(!list.children.length)list.appendChild(el('p','No matching candidates.','empty'))}
function renderDetail(){const root=$('detail');root.replaceChildren();const c=data.candidates.find(c=>c.id===selected);if(!c){panel('No saved candidates',root).appendChild(el('p','This snapshot has no candidate circuits yet.'));return}const intro=panel(null,root);intro.appendChild(el('div',c.purpose==='verification'?'Fresh verification evaluation':'Candidate circuit','eyebrow'));intro.appendChild(el('h2',c.name));intro.appendChild(el('small','Evaluation '+c.id));const labels=el('div',undefined,'badges');badge(labels,c.status);badge(labels,feasibility(c),c.score.feasible===true?'pass':c.score.feasible===false?'fail':'unknown');badge(labels,c.score.valid===true?'valid evaluation':c.score.valid===false?'invalid evaluation':'validity unknown');if(c.id===data.best_id)badge(labels,data.analysis_only?'Selected for remeasurement':'Selected search winner');if(c.purpose==='verification')badge(labels,c.verified?'Fresh verification recorded':'Fresh verification not confirmed');intro.appendChild(labels);const stats=el('div',undefined,'stats');[['Search rank',c.rank??'unranked'],['Physical packages',c.score.package_count],['Worst normalized violation',c.worst],['Evaluation',c.purpose==='verification'?'fresh':'baseline / search']].forEach(([name,value])=>{const stat=el('div',undefined,'stat');stat.appendChild(el('small',name));stat.appendChild(el('strong',fmt(value)));stats.appendChild(stat)});intro.appendChild(stats);intro.appendChild(el('p',data.analysis_only?'Not scored: no acceptance requirements were frozen. Measurements available; start a new experiment with explicit bounds to rank alternatives.':'All required checks take priority over component count. Rank is the saved server order or the same recorded lexicographic rank_key; smaller keys rank first.','notice'));intro.appendChild(el('p','Recorded rank_key: '+fmt(c.score.rank_key),'muted'));if(Object.hasOwn(c.score,'points'))intro.appendChild(el('p','Legacy recorded points: '+fmt(c.score.points)+' (informational; rank_key controls order)','muted'));if(data.selection_policy)intro.appendChild(el('p',data.selection_policy,'muted'));c.errors.forEach(error=>intro.appendChild(el('p',error,'fail')));const tabs=el('nav',undefined,'tabs');tabs.setAttribute('aria-label','Candidate details');[['circuit','Circuit & BOM'],['outputs','Output plots'],['kicad','KiCad files'],['metrics','Measurements & checks'],['compare','Compare evaluations'],['evidence','SPICE & evidence']].forEach(([id,label])=>{const b=el('button',label,'tab');b.type='button';b.setAttribute('aria-selected',activeTab===id);b.onclick=()=>{activeTab=id;renderDetail()};tabs.appendChild(b)});root.appendChild(tabs);
if(activeTab==='circuit'){const box=panel('Circuit reconstructed from this candidate’s saved IR',root),drawing=el('div',undefined,'schematic');drawing.innerHTML=c.svg;const tools=el('div',undefined,'tabs');[['Fit width','100%'],['Actual size','auto'],['Zoom 125%','125%']].forEach(([label,width])=>{const button=el('button',label,'tab');button.type='button';button.onclick=()=>{drawing.querySelector('svg').style.width=width};tools.appendChild(button)});box.appendChild(tools);box.appendChild(drawing);const parts=c.circuit.components||[];const bom=panel('Bill of materials · required external components',root);bom.appendChild(el('p','Physical package grouping uses the saved package field (or reference when absent). The authoritative count above is the recorded evaluator count.','muted'));bom.appendChild(table(['Reference','Kind / model','Value (SI)','Nodes','Physical package','Role'],parts.map(p=>[p.ref,[p.kind,p.model].filter(Boolean).join(' / '),p.value,(p.nodes||[]).join(' → '),p.package||p.ref,p.role||''])));const groups=new Map();parts.forEach(p=>{const key=(p.package||p.ref||'unknown').toLowerCase();if(!groups.has(key))groups.set(key,[]);groups.get(key).push(p.ref)});bom.appendChild(el('h3','Physical package membership'));bom.appendChild(table(['Package','Channels / references'],[...groups].map(([key,refs])=>[key,refs.join(', ')])))}
if(activeTab==='outputs'){const box=panel('Saved output curves · this evaluation only',root);box.appendChild(el('p',c.purpose==='verification'?'These curves belong to the fresh verification evaluation of the selected circuit.':'These curves belong to this saved search evaluation. Select the fresh verification row to inspect its independent remeasurement.','muted'));if(!c.outputs.length){box.appendChild(el('p','Output plots unavailable: no readable saved AC or transient samples are available in the authorized workspace artifact folder. No curves were synthesized.','empty'))}else{const label=el('label','Choose analysis and probe');label.htmlFor='plot-choice';box.appendChild(label);const choice=el('select',undefined,'search');choice.id='plot-choice';c.outputs.forEach((plot,index)=>{const option=el('option',plot.title);option.value=index;choice.appendChild(option)});box.appendChild(choice);const target=el('div'),note=el('p',undefined,'muted');target.style.overflow='auto';box.appendChild(target);box.appendChild(note);const show=()=>{const plot=c.outputs[Number(choice.value)];target.innerHTML=plot.svg;const svg=target.querySelector('svg');if(svg){svg.style.width='100%';svg.style.minWidth='620px';svg.style.height='auto'}note.textContent=plot.note};choice.onchange=show;show()}}
if(activeTab==='kicad'){const box=panel('Native KiCad files',root);box.appendChild(el('p','Open the project in KiCad, then its schematic. Keep the exported models in place. File links open manually; browser support varies.','muted'));box.appendChild(el('p','Saved scores apply to the original Circuit IR. Manual edits require reevaluation.','notice'));if(!c.kicad||!c.kicad.files.length){box.appendChild(el('p','No existing native export files are available for this candidate. Create a bundle in the terminal, then refresh this viewer.'));box.appendChild(el('pre',data.export_command))}if(c.kicad){box.appendChild(el('p',c.kicad.native_runnable&&c.kicad.status==='ok'&&!c.kicad.missing.length?'Native simulation dependencies recorded as available.':'Native simulation is not confirmed ready. Export status: '+c.kicad.status,'muted'));c.kicad.missing.forEach(message=>box.appendChild(el('p',fmt(message),'unknown')));c.kicad.files.forEach(file=>{const line=el('p'),link=el('a',file.name);link.href=file.href;link.rel='noopener noreferrer';line.appendChild(link);line.appendChild(el('pre',file.path));box.appendChild(line)})}}
if(activeTab==='metrics'){const box=panel('Measured constraints and analysis-only metrics',root);box.appendChild(el('p','Normalized violations and pass/fail are recorded evaluator results. Missing evidence remains unknown. An individual passed check does not make an invalid evaluation valid.','muted'));box.appendChild(table(['Metric / scope','Measured value','Frozen bound','Normalized violation','Recorded check'],c.metrics.map(m=>[m.metric+' / '+m.scope,m.value,m.bound,m.violation,el('span',m.state,m.state)])));if(!c.metrics.length)box.appendChild(el('p','No measurements or constraints have been saved.','empty'))}
if(activeTab==='compare'){const best=data.candidates.find(v=>v.id===data.best_id)||data.candidates.find(v=>v.rank===1);const verify=data.candidates.find(v=>v.id===data.verification_id);const box=panel('Baseline, selection and fresh verification',root);box.appendChild(el('p','Fresh verification is a separate evaluation of the selected circuit, not another search competitor.','muted'));const columns=[c];if(best&&best.id!==c.id)columns.push(best);if(verify&&!columns.some(v=>v.id===verify.id))columns.push(verify);box.appendChild(table(['Evaluation','Purpose','Feasibility','Packages','Worst violation','Recorded rank_key'],columns.map(v=>[v.name+' · '+v.id,v.purpose,feasibility(v),v.score.package_count,v.worst,fmt(v.score.rank_key)])));const metricBox=panel('Side-by-side recorded measurements',root);const names=[...new Set(columns.flatMap(v=>v.metrics.map(m=>m.scope+' / '+m.metric)))];metricBox.appendChild(table(['Metric',...columns.map(v=>v.name+' · '+(v.purpose==='verification'?'fresh verification':v.id===data.best_id?'selected search':v.id.slice(0,8)))],names.map(name=>[name,...columns.map(v=>v.metrics.find(m=>m.scope+' / '+m.metric===name)?.value)])))}
if(activeTab==='evidence'){const box=panel('Generated SPICE decks & local evidence',root);box.appendChild(el('p','Only saved artifacts inside the authorized workspace artifact folder are linked. This page never executes decks. Browser file-link support varies.','muted'));if(!c.artifacts.length)box.appendChild(el('p','No authorized local artifact files are available in this snapshot.','empty'));c.artifacts.forEach(a=>{const line=el('p'),link=el('a',a.name);link.href=a.href;link.rel='noopener noreferrer';line.appendChild(link);const location=el('small',a.path);location.style.display='block';location.style.overflowWrap='anywhere';line.appendChild(location);box.appendChild(line);if(a.deck!==undefined){const d=el('details');d.appendChild(el('summary','View generated deck · '+a.name));d.appendChild(el('pre',a.deck));box.appendChild(d)}});const meta=panel('Saved evidence metadata',root);meta.appendChild(el('pre',JSON.stringify(c.fingerprints,null,2)));const spec=panel('Frozen specification / provenance',root);spec.appendChild(el('pre',JSON.stringify(data.spec,null,2)));const ir=panel('Selected candidate’s saved Circuit IR',root);ir.appendChild(el('pre',JSON.stringify(c.circuit,null,2)))}
}
$('run-info').textContent='Run '+data.run_id;badge($('run-badges'),data.status);badge($('run-badges'),data.mode+(data.offline?' · no agents':''));badge($('run-badges'),data.verified?'Fresh verification recorded':'Fresh verification not confirmed',data.verified?'pass':'unknown');badge($('run-badges'),data.feasible===true?(data.verified?'Verified feasible':'Result feasible (unverified)'):data.feasible===false?'Result infeasible':data.analysis_only?'Analysis only':'Feasibility unknown',data.feasible===true?'pass':data.feasible===false?'fail':'unknown');$('snapshot').textContent='Saved state: '+(data.updated_at||'time unavailable')+' · HTML snapshot: '+data.snapshot_at+'. Static snapshot; refresh by exporting again: '+data.refresh_command;$('candidate-count').textContent=data.candidates.filter(c=>c.purpose!=='verification').length+' candidate circuits';const freshCount=data.candidates.filter(c=>c.purpose==='verification').length;$('candidate-count').appendChild(el('small',' + '+freshCount+' fresh remeasurement'+(freshCount===1?'':'s')));$('search').addEventListener('input',renderList);renderList();renderDetail();
</script></body></html>'''
