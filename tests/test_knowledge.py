"""Local retrieval tests, with no network or provider calls."""
import hashlib
from pathlib import Path

import pytest
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from spiceharness.knowledge import RECIPES, KnowledgeStore


def test_original_seed_provenance_and_idempotence(tmp_path: Path):
    knowledge = KnowledgeStore(tmp_path / "knowledge.sqlite3")
    initial = knowledge.stats()
    assert initial["documents"] == len(RECIPES)
    knowledge.seed()
    assert knowledge.stats() == initial
    matches = knowledge.search("passive_lowpass", limit=3)
    assert matches
    assert matches[0]["family"] == "passive_lowpass"
    assert all(row["kind"] == "original_recipe_note" for row in matches)
    assert all(row["source"].startswith("https://www.allaboutcircuits.com/textbook/") for row in matches)
    assert all(len(row["checksum"]) == 64 for row in matches)


def test_local_import_dedup_and_fts_escaping(tmp_path: Path):
    document = tmp_path / "design_notes.md"
    document.write_text("# Banana amplifier\nbanana bias current is measured at the collector.\n")
    knowledge = KnowledgeStore(tmp_path / "knowledge.sqlite3")
    imported = knowledge.import_document(document)
    assert imported["status"] == "imported"
    assert knowledge.import_document(document)["status"] == "already_imported"
    found = knowledge.search('banana" OR * NOT (')
    assert found
    assert found[0]["source"] == str(document.resolve())
    assert found[0]["kind"] == "local_text"
    assert knowledge.search("   ") == []
    document.write_text("Updated banana bias notes.")
    assert knowledge.import_document(document)["status"] == "imported"


def test_powerline_noise_recipes_are_searchable_with_provenance(tmp_path: Path):
    knowledge = KnowledgeStore(tmp_path / "knowledge.sqlite3")
    matches = knowledge.search("powerline noise", limit=3)
    relevant = [row for row in matches if "powerline_noise" in row["family"].split()]
    assert relevant
    for row in relevant:
        assert row["kind"] == "original_recipe_note"
        assert row["source"].startswith("https://www.ti.com/")
        assert row["checksum"] == hashlib.sha256(row["content"].encode()).hexdigest()
        assert row["page"] is None


def test_pdf_text_pages_and_scan_warning(tmp_path: Path):
    writer = PdfWriter()
    page = writer.add_blank_page(width=300, height=300)
    stream = DecodedStreamObject()
    stream.set_data(b"BT /F1 12 Tf 20 250 Td (Zephyr oscillator measured gain reference text.) Tj ET")
    page[NameObject("/Contents")] = writer._add_object(stream)
    font = DictionaryObject({NameObject("/Type"): NameObject("/Font"),
                             NameObject("/Subtype"): NameObject("/Type1"),
                             NameObject("/BaseFont"): NameObject("/Helvetica")})
    page[NameObject("/Resources")] = DictionaryObject({NameObject("/Font"):
        DictionaryObject({NameObject("/F1"): writer._add_object(font)})})
    writer.add_blank_page(width=300, height=300)
    pdf = tmp_path / "reference.pdf"
    writer.write(pdf)
    knowledge = KnowledgeStore(tmp_path / "knowledge.sqlite3")
    imported = knowledge.import_document(pdf)
    assert imported["status"] == "imported"
    assert imported["pages_requiring_ocr"] == [2]
    match = knowledge.search("Zephyr")[0]
    assert match["page"] == 1
    assert match["kind"] == "local_text_pdf"
    assert match["source"] == str(pdf.resolve())


def test_scan_only_pdf_is_not_claimed_as_indexed(tmp_path: Path):
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    pdf = tmp_path / "scan.pdf"
    writer.write(pdf)
    knowledge = KnowledgeStore(tmp_path / "knowledge.sqlite3")
    imported = knowledge.import_document(pdf)
    assert imported["status"] == "requires_ocr"
    assert imported["chunks"] == 0
    assert "OCR" in imported["warning"]


def test_corrupt_pdf_and_empty_text_are_reported_cleanly(tmp_path: Path):
    knowledge = KnowledgeStore(tmp_path / "knowledge.sqlite3")
    corrupt = tmp_path / "corrupt.pdf"
    corrupt.write_bytes(b"%PDF-1.4 garbage")
    # A pypdf stream error used to escape as a raw traceback instead of a ValueError.
    with pytest.raises(ValueError, match="Could not read PDF corrupt.pdf"):
        knowledge.import_document(corrupt)
    empty = tmp_path / "empty.md"
    empty.write_text("  \n")
    result = knowledge.import_document(empty)
    # The OCR hint applies to scanned PDFs only.
    assert result["status"] == "empty"
    assert "OCR" not in result["warning"]
