"""Local FTS5 retrieval with original recipe notes and explicit provenance."""
from __future__ import annotations

import hashlib
import re
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .storage import utcnow

# These are original summaries, not copies of the cited textbook. Sources are
# provenance for the circuit principles, not a claim of a licensed scraped corpus.
RECIPES = [
    {"title": "Non-inverting op-amp gain and headroom", "family": "noninverting_amplifier",
     "source": "https://www.allaboutcircuits.com/textbook/semiconductors/chpt-8/negative-feedback/",
     "content": "A non-inverting amplifier starts from gain 1 + Rf/Rg. Pick the resistor ratio analytically, then try a few standard-value resistance scales. A real op-amp needs valid input common-mode voltage, supply rails and output headroom; the ideal gain equation alone does not establish feasibility. Check operating point, loaded AC response and transient clipping with the actual model."},
    {"title": "Inverting op-amp gain and source loading", "family": "inverting_amplifier",
     "source": "https://www.allaboutcircuits.com/textbook/semiconductors/chpt-8/negative-feedback/",
     "content": "An inverting amplifier uses gain -Rf/Rin for an ideal low-impedance source. Include source resistance in the effective input resistance and check the load in simulation. Lower resistor values reduce the impact of bias current but increase current demand. A single positive supply may require a bias reference; do not silently assume a grounded input can swing below the negative rail."},
    {"title": "Passive RC low-pass including source and load", "family": "passive_lowpass",
     "source": "https://www.allaboutcircuits.com/textbook/alternating-current/chpt-8/low-pass-filters/",
     "content": "For an unloaded RC low-pass, seed C from 1/(2*pi*R*fc). In a loaded network the capacitor sees (R + source resistance) in parallel with the load; use that resistance to estimate the pole. The DC divider gain is load/(load + R + source resistance). A cutoff target and an insertion-gain target can conflict. Try standard capacitors and solve the resistor analytically before SPICE refinement."},
    {"title": "Buffered first-order active low-pass", "family": "active_lowpass_1",
     "source": "https://www.allaboutcircuits.com/textbook/semiconductors/chpt-8/operational-amplifiers/",
     "content": "A first-order RC stage followed by a non-inverting op-amp separates the passive pole from output loading when the amplifier has adequate input impedance and bandwidth. Seed the pole with RC and the gain with the feedback ratio. Validate finite gain-bandwidth, slew rate, common-mode limits and loaded output swing. This is a first-order family and must not be advertised as a higher-order response."},
    {"title": "Common-emitter bias and degeneration", "family": "common_emitter",
     "source": "https://www.allaboutcircuits.com/textbook/semiconductors/chpt-4/common-emitter-amplifier/",
     "content": "Start a BJT common-emitter stage from a collector quiescent voltage near the middle of the available output range. Select emitter current and emitter degeneration before solving the base divider and collector resistor. A rough midband gain is minus the effective collector load divided by emitter resistance plus intrinsic emitter resistance. Base current loads the bias divider. Verify active-region bias, coupling-capacitor poles, load, distortion and clipping with a transistor model."},
    {"title": "Passive RC high-pass and DC blocking", "family": "passive_highpass",
     "source": "https://www.allaboutcircuits.com/textbook/alternating-current/chpt-8/high-pass-filters/",
     "content": "An RC high-pass places a series capacitor before a shunt resistance and rejects DC. Include source resistance and the parallel combination of the shunt resistor and load when seeding its pole: C = 1/(2*pi*fc*(Rs + (R parallel RL))). The high-frequency gain includes a resistive divider. A coupling capacitor does not supply a missing DC bias path; verify operating point as well as AC and transient response."},
    {"title": "Unity-gain op-amp buffer", "family": "buffer",
     "source": "https://www.allaboutcircuits.com/textbook/semiconductors/chpt-8/voltage-follower/",
     "content": "A voltage follower feeds output directly to the inverting input and drives the non-inverting input with the signal. It provides impedance conversion with nominal unity gain. The selected device must tolerate unity-gain operation, the input common-mode range, the required output current and the capacitive load. Count the physical op-amp package and any bias and supply decoupling components. Verify real loaded gain, distortion and rail headroom."},
    {"title": "Emitter follower bias and loaded gain", "family": "emitter_follower",
     "source": "https://www.allaboutcircuits.com/textbook/semiconductors/chpt-4/common-collector-amplifier/",
     "content": "A common-collector emitter follower has voltage gain somewhat below one and can isolate a source from a heavier load. The base needs a bias about one junction voltage above the desired emitter DC voltage. Bias-divider loading and emitter current determine useful input impedance and output swing. A single transistor cannot provide arbitrary symmetric current drive. Include coupling and bias parts in the count and simulate loaded transient behavior."},
    {"title": "Buffered first-order active high-pass", "family": "active_highpass_1",
     "source": "https://www.allaboutcircuits.com/textbook/alternating-current/chpt-8/high-pass-filters/",
     "content": "Buffering an RC high-pass can reduce output-load influence on the pole. Seed the capacitor from the resistance it sees, then check the op-amp input bias path, common-mode voltage and upper bandwidth. Any intended passband gain requires actual feedback components. Validate both the lower high-pass cutoff and the upper device bandwidth; a high-pass label does not mean infinite high-frequency gain."},
    {"title": "Two buffered poles and total filter cutoff", "family": "active_lowpass_2 active_highpass_2",
     "source": "https://www.allaboutcircuits.com/textbook/alternating-current/chpt-8/low-pass-filters/",
     "content": "Two buffered first-order stages multiply their transfer functions and give a second-order asymptotic slope. Their individual pole frequencies must be adjusted to meet a specified total minus-three-decibel cutoff. For equal low-pass poles use individual fp = fc/sqrt(sqrt(2)-1); for equal high-pass poles use fp = fc*sqrt(sqrt(2)-1). This cascade is not a Butterworth response. Count each external RC stage and the physical multi-amplifier package, and verify requested stopband attenuation."},
    {"title": "General experiment reconstruction and frozen fixtures", "family": "general_experiment",
     "source": "https://ngspice.sourceforge.io/docs/ngspice-manual.pdf",
     "content": "Reconstruct a problem as explicit nodes, component values and units, source impedance, load, supply rails, probes and analysis conditions before comparing circuits. Record ambiguous connections or noise definitions as assumptions; do not invent equivalence to another simulator. Freeze the specification, fixtures and protected sections, then simulate a baseline through validated Circuit IR. Retrieved notes may suggest candidates, but only the deterministic SPICE evaluator establishes measurements and numerical ranking. Sensitivity samples are not exhaustive worst-case evidence."},
    {"title": "Supply RC, LC, T and pi filters: resonance and damping", "family": "supply_filter powerline_noise general_experiment",
     "source": "https://www.ti.com/lit/pdf/snva538",
     "content": "For powerline noise, compare RC, LC, T and pi supply filters under the same source impedance and load. RC filtering trades attenuation against DC drop and resistor dissipation. Use 1/(2*pi*sqrt(L*C)) only as an initial LC resonance estimate; source and load impedance, winding resistance, capacitor ESR and extra branches change peaking and attenuation. A T or pi topology does not guarantee better loaded response. Consider resistive or series-RC shunt damping, count its parts, and verify resonance, DC delivery and load transients in SPICE. A converter load may require input-impedance and control-loop stability evidence beyond a resistor-load test."},
    {"title": "External powerline noise PSD, PSRR and bandwidth limits", "family": "powerline_noise supply_filter general_experiment",
     "source": "https://www.ti.com/document-viewer/lit/html/SSZTCB3/GUID-78BC003E-2886-49F6-A1C5-C592550413B6",
     "content": "Define external powerline noise separately from internal device noise and deterministic ripple. State injection node, reference, PSD units and integration limits. For a linear small-signal transfer H, output noise variance is the integral of input PSD times abs(H)^2 over the stated band; amplitude spectral density must first be squared. PSRR varies with frequency and operating conditions, so a single rejection number cannot predict broadband noise. Integrate only within supported model and analysis bandwidth, and do not assume a macromodel includes measured PSRR or internal noise. Retrieved context guides investigation; evaluator-owned SPICE results establish the reported evidence."},
]


class KnowledgeStore:
    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY, title TEXT NOT NULL, source TEXT NOT NULL,
                    checksum TEXT NOT NULL, kind TEXT NOT NULL, imported_at TEXT NOT NULL,
                    UNIQUE(source, checksum)
                );
                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY, document_id INTEGER NOT NULL REFERENCES documents(id),
                    chunk_index INTEGER NOT NULL, page INTEGER, family TEXT NOT NULL DEFAULT '',
                    content TEXT NOT NULL, UNIQUE(document_id, chunk_index)
                );
                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                    title, family, content, content='');
            """)
        self.seed()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.path, timeout=15)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA foreign_keys=ON")
        try:
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def _insert(self, *, title: str, source: str, checksum: str, kind: str,
                chunks: list[tuple[int | None, str]], family: str = "") -> dict[str, Any]:
        with self._connect() as db:
            found = db.execute("SELECT id FROM documents WHERE source=? AND checksum=?",
                               (source, checksum)).fetchone()
            if found:
                return {"document_id": found["id"], "status": "already_imported", "chunks": 0,
                        "source": source, "checksum": checksum}
            cursor = db.execute("INSERT INTO documents(title,source,checksum,kind,imported_at) VALUES(?,?,?,?,?)",
                                (title, source, checksum, kind, utcnow()))
            document_id = cursor.lastrowid
            for index, (page, content) in enumerate(chunks):
                cursor = db.execute("INSERT INTO chunks(document_id,chunk_index,page,family,content) VALUES(?,?,?,?,?)",
                                    (document_id, index, page, family, content))
                db.execute("INSERT INTO chunks_fts(rowid,title,family,content) VALUES(?,?,?,?)",
                           (cursor.lastrowid, title, family, content))
        return {"document_id": document_id, "status": "imported", "chunks": len(chunks),
                "source": source, "checksum": checksum}

    def seed(self) -> None:
        for recipe in RECIPES:
            self._insert(title=recipe["title"], source=recipe["source"],
                         checksum=hashlib.sha256(recipe["content"].encode()).hexdigest(),
                         kind="original_recipe_note", chunks=[(None, recipe["content"])], family=recipe["family"])

    @staticmethod
    def _chunks(text: str, size: int = 1600) -> list[str]:
        text = re.sub(r"[ \t]+", " ", text).strip()
        if not text:
            return []
        result: list[str] = []
        while text:
            stop = min(len(text), size)
            if stop < len(text):
                boundary = text.rfind(" ", size // 2, size)
                if boundary > 0:
                    stop = boundary
            result.append(text[:stop].strip())
            text = text[stop:].lstrip()
        return result

    def import_document(self, path: str | Path) -> dict[str, Any]:
        path = Path(path).expanduser().resolve(strict=True)
        if not path.is_file() or path.suffix.lower() not in {".md", ".txt", ".pdf"}:
            raise ValueError("Knowledge import accepts local .md, .txt, or text-based .pdf files")
        if path.stat().st_size > 25 * 1024 * 1024:
            raise ValueError("Knowledge documents must be at most 25 MiB")
        content = path.read_bytes()
        checksum = hashlib.sha256(content).hexdigest()
        chunks: list[tuple[int | None, str]] = []
        empty_pages: list[int] = []
        if path.suffix.lower() == ".pdf":
            from pypdf import PdfReader
            from pypdf.errors import PyPdfError
            try:
                reader = PdfReader(path)
                if reader.is_encrypted:
                    raise ValueError("Encrypted PDFs must be decrypted before knowledge import")
                if len(reader.pages) > 1000:
                    raise ValueError("PDF knowledge import is limited to 1000 pages")
                for number, page in enumerate(reader.pages, 1):
                    text = page.extract_text() or ""
                    if len(text.strip()) < 20:
                        empty_pages.append(number)
                    chunks.extend((number, part) for part in self._chunks(text))
            except PyPdfError as exc:
                raise ValueError(f"Could not read PDF {path.name}: {exc}") from exc
            kind = "local_text_pdf"
        else:
            text = content.decode("utf-8-sig")
            chunks = [(None, part) for part in self._chunks(text)]
            kind = "local_text"
        if not chunks:
            is_pdf = path.suffix.lower() == ".pdf"
            return {"status": "requires_ocr" if is_pdf else "empty",
                    "source": str(path), "checksum": checksum, "chunks": 0,
                    "warning": ("No extractable text; scanned PDFs require external OCR before import."
                                if is_pdf else "The file contains no text to import.")}
        result = self._insert(title=path.stem, source=str(path), checksum=checksum,
                              kind=kind, chunks=chunks)
        if empty_pages:
            result["pages_requiring_ocr"] = empty_pages
            result["warning"] = "Some pages have little or no text; image content was not indexed."
        return result

    import_path = import_document

    def search(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        if not 1 <= limit <= 50:
            raise ValueError("Search limit must be between 1 and 50")
        # All FTS syntax is constructed here: user text cannot become FTS operators.
        tokens = re.findall(r"[\w]+", query.lower(), flags=re.UNICODE)[:24]
        if not tokens:
            return []
        match = " OR ".join('"' + token.replace('"', '""') + '"' for token in tokens)
        with self._connect() as db:
            rows = db.execute("""
                SELECT c.id AS chunk_id,c.content,c.page,c.family,d.title,d.source,d.checksum,d.kind,
                       bm25(chunks_fts, 3.0, 4.0, 1.0) AS relevance
                FROM chunks_fts JOIN chunks c ON c.id=chunks_fts.rowid
                JOIN documents d ON d.id=c.document_id WHERE chunks_fts MATCH ?
                ORDER BY relevance,c.id LIMIT ?
            """, (match, limit)).fetchall()
        return [dict(row) for row in rows]

    def stats(self) -> dict[str, int]:
        with self._connect() as db:
            return {"documents": db.execute("SELECT COUNT(*) FROM documents").fetchone()[0],
                    "chunks": db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]}
