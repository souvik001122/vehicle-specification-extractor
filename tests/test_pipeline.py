"""End-to-end and unit tests. Run with `python tests/test_pipeline.py` or `pytest -q` (offline, no API key)."""
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from src.extractor import Extractor, parse_llm_json, public, value_in_text, table_records  # noqa: E402
from src.chunking import Chunk  # noqa: E402
from src.main import Pipeline, write_output  # noqa: E402
from src.textutil import parse_query  # noqa: E402

_PIPE = None


def pipe() -> Pipeline:
    """Build the index once (about a second with the TF-IDF backend)."""
    global _PIPE
    if _PIPE is None:
        _PIPE = Pipeline(embed_backend="tfidf", extract_backend="rules")
    return _PIPE


def rows(query: str) -> list[dict]:
    recs, _, _ = pipe().run(query)
    return [public(r) for r in recs]


def has(rs, component_part, value, unit=None, section_part=None):
    return any(component_part.lower() in r["component"].lower() and r["value"] == value
               and (unit is None or r["unit"] == unit)
               and (section_part is None or section_part in r["section"]) for r in rs)


class TestQueryParsing(unittest.TestCase):
    def test_types(self):
        self.assertEqual(parse_query("Torque for brake caliper bolts").spec_type, "torque")
        self.assertEqual(parse_query("Rear axle lubricant capacity").spec_type, "capacity")
        self.assertEqual(parse_query("Part number for shock absorber").spec_type, "part_number")
        self.assertEqual(parse_query("Brake disc minimum thickness").spec_type, "measurement")

    def test_keywords(self):
        self.assertEqual(parse_query("Torque for brake caliper bolts").keywords, ["brake", "caliper", "bolt"])


class TestIngestion(unittest.TestCase):
    def test_cleaning(self):
        for d in pipe().docs:
            for ln in d.lines:
                self.assertNotIn("file:///", ln)
                self.assertNotRegex(ln, r"Page \d+ sur \d+")

    def test_sections_and_chunks(self):
        p = pipe()
        self.assertGreater(len(p.docs), 100)
        self.assertGreater(len(p.chunks), 300)
        self.assertTrue(all(c.section_id for c in p.chunks))
        self.assertTrue(any(c.kind == "torque_table" for c in p.chunks))
        self.assertTrue(any(c.kind == "material_table" for c in p.chunks))
        self.assertTrue(all(len(c.text) < config.MAX_CHUNK_CHARS * 4 for c in p.chunks))


class TestExtraction(unittest.TestCase):
    """Ground truth read by hand from the manual's specification tables."""

    def test_caliper_bolts_front_and_rear_differ(self):
        rs = rows("Torque for brake caliper bolts")
        # Check every relevant row, not just a subset: front and rear values differ.
        expected = [
            ("anchor plate", "250", "206-03"),
            ("flow bolt", "35", "206-03"),
            ("guide pin", "37", "206-03"),
            ("flow bolt", "35", "206-04"),
            ("guide pin", "33", "206-04"),
            ("support bracket", "150", "206-04"),
        ]
        for component, value, section in expected:
            self.assertTrue(has(rs, component, value, "Nm", section),
                            msg=f"Missing {component} = {value} Nm in {section}; got {rs}")
        self.assertEqual(sum(1 for r in rs if r["spec_type"] == "Torque"), 6,
                         msg=f"Expected six brake-caliper torque rows, got {rs}")

    def test_caliper_query_prioritizes_direct_evidence(self):
        _, hits, _ = pipe().run("Torque for brake caliper bolts")
        sections = [chunk.section for chunk, _ in hits]
        self.assertTrue(any("206-03" in section for section in sections))
        self.assertTrue(any("206-04" in section for section in sections))
        self.assertFalse(any("206-05" in section or "206-07" in section or "205-04" in section
                             for section in sections))

    def test_wheel_nuts(self):
        self.assertTrue(has(rows("Torque for wheel nuts"), "Wheel nuts", "204", "Nm"))

    def test_shock_and_stabilizer(self):
        self.assertTrue(has(rows("Torque for shock absorber lower nut"), "shock absorber lower nut", "90", "Nm"))
        self.assertTrue(has(rows("Torque for stabilizer bar link nuts"), "Stabilizer bar link nuts", "70", "Nm"))

    def test_skid_plate(self):
        self.assertTrue(has(rows("Torque for skid plate bolts"), "Skid plate bolts", "48", "Nm"))

    def test_capacity(self):
        self.assertTrue(has(rows("Brake fluid capacity"), "Brake Fluid", "740", "ml"))
        self.assertTrue(has(rows("Rear axle lubricant capacity"), "Rear Axle Lubricant", "2.84", "L"))

    def test_part_number(self):
        self.assertTrue(has(rows("Part number for shock absorber"), "Shock absorber lower bolt", "W506545"))

    def test_measurement(self):
        rs = rows("Brake disc minimum thickness")
        self.assertTrue(has(rs, "Front brake disc", "32", "mm"))
        self.assertTrue(has(rs, "Rear brake disc", "23.0", "mm"))

    def test_pdf_flattened_torque_table_is_parsed_from_text_chunk(self):
        # PDF text extraction can put a table row's name, values, and dash cells on separate lines.
        chunk = Chunk(
            id=9999, doc_index=0, section_id="206-03", section_title="Front Disc Brake",
            mode="SPECIFICATIONS", heading="General Specifications", kind="text",
            text="""Description Nm lb-ft lb-in
Brake caliper anchor plate bolts
250 184
—
Brake caliper flow bolt
35 26
—
Brake caliper guide pin bolts
37 27
—
Item Specification""",
        )
        records = table_records(chunk)
        self.assertTrue(any(r.component.lower() == "brake caliper guide pin bolts" and r.value == "37" and r.unit == "Nm" for r in records))
        self.assertTrue(any(r.component.lower() == "brake caliper anchor plate bolts" and r.value == "250" and r.unit == "Nm" for r in records))

    def test_pdf_split_torque_header_and_rows_are_parsed_from_text_chunk(self):
        # Some PDF parsers emit the table header labels and each cell on separate lines.
        chunk = Chunk(
            id=10000, doc_index=0, section_id="206-03", section_title="Front Disc Brake",
            mode="SPECIFICATIONS", heading="General Specifications", kind="text",
            text="""Description
Nm
lb-ft
lb-in
Brake caliper anchor plate bolts
250 184
—
Brake caliper flow bolt
35
26
—
Brake caliper guide pin bolts
37
27
—""",
        )
        records = table_records(chunk)
        self.assertTrue(any(r.component.lower() == "brake caliper anchor plate bolts" and r.value == "250" and r.unit == "Nm" for r in records), records)
        self.assertTrue(any(r.component.lower() == "brake caliper flow bolt" and r.value == "35" and r.unit == "Nm" for r in records), records)
        self.assertTrue(any(r.component.lower() == "brake caliper guide pin bolts" and r.value == "37" and r.unit == "Nm" for r in records), records)

    def test_procedure_torque_is_extracted(self):
        chunk = Chunk(
            id=10001, doc_index=0, section_id="204-00", section_title="Suspension System — General Information",
            mode="GENERAL PROCEDURES", heading="Toe Adjustment — Front", kind="text",
            text="Tighten the tie-rod end jam nut(s) to 103 Nm (76 lb-ft).",
        )
        from src.extractor import procedure_records
        records = procedure_records(chunk, set())
        self.assertTrue(any(r.value == "103" and r.unit == "Nm" for r in records), records)

    def test_embedded_spec_table_is_parsed(self):
        chunk = Chunk(
            id=10002, doc_index=0, section_id="206-00", section_title="Brake System — General Information",
            mode="SPECIFICATIONS", heading="General Specifications", kind="text",
            text="Item Specification\nFront brake disc minimum thickness 32 mm (1.259 in)\nRear brake disc minimum thickness 23.0 mm (0.906 in)",
        )
        records = table_records(chunk)
        self.assertTrue(any(r.component == "Front brake disc" and r.value == "32" and r.unit == "mm" for r in records), records)

    def test_nonsense_returns_nothing(self):
        self.assertEqual(rows("Torque for flux capacitor bolts"), [])

    def test_output_schema(self):
        for r in rows("Torque for wheel nuts"):
            self.assertEqual(set(r), {"component", "spec_type", "value", "unit", "section"})
            self.assertEqual(r["spec_type"], "Torque")
        self.assertEqual(set(public({"component": "a", "spec_type": "b", "value": "1", "unit": "Nm"}, minimal=True)),
                         {"component", "spec_type", "value", "unit"})


class TestLLMPath(unittest.TestCase):
    """The LLM backend is tested with a fake model so no API key is needed."""

    def _run(self, reply):
        p = pipe()
        ex = Extractor("rules", llm_fn=lambda prompt: reply)
        hits = p.retriever.search("Torque for wheel nuts")
        return ex.extract("Torque for wheel nuts", hits)

    def test_valid_reply_is_used(self):
        recs, used = self._run('{"specs":[{"component":"Wheel nuts","spec_type":"Torque","value":"204","unit":"Nm","chunk":1}]}')
        self.assertEqual(used, "custom")
        self.assertEqual(recs[0].value, "204")

    def test_hallucinated_value_is_dropped_and_rules_take_over(self):
        recs, used = self._run('```json\n{"specs":[{"component":"Wheel nuts","spec_type":"Torque","value":"999","unit":"Nm"}]}\n```')
        self.assertIn("rules", used)
        self.assertTrue(any(r.value == "204" for r in recs))

    def test_broken_json_falls_back(self):
        _, used = self._run("sorry, I cannot")
        self.assertIn("rules", used)

    def test_helpers(self):
        self.assertTrue(value_in_text("1,420", "fill 1,420 ml"))
        self.assertFalse(value_in_text("14", "fill 1,420 ml"))
        self.assertEqual(len(parse_llm_json('[{"a":1}]')), 1)


class TestOutputFiles(unittest.TestCase):
    def test_json_and_csv(self):
        import tempfile
        res = [("q", rows("Torque for wheel nuts"))]
        with tempfile.TemporaryDirectory() as d:
            write_output(Path(d) / "o.json", res, single=True)
            self.assertIsInstance(json.loads((Path(d) / "o.json").read_text()), list)
            write_output(Path(d) / "o.csv", res, single=False)
            self.assertTrue((Path(d) / "o.csv").read_text().startswith("query,component"))


class TestNeverZeroRegression(unittest.TestCase):
    """Bug: with sbert embeddings the spec table fell outside the top-k and every query returned 0 rows."""

    def test_query_word_missing_from_table(self):
        # the capacity table never says "drive"; only its section title does
        self.assertTrue(has(rows("Front drive axle lubricant capacity"), "Axle Lubricant", "1.66", "L"))

    def test_fallback_when_retrieval_returns_only_text_chunks(self):
        p = pipe()
        real = p.retriever.search
        try:
            p.retriever.search = lambda q, k=None: [(c, 0.0) for c in p.chunks if not c.is_table][: (k or 8)]
            recs, _, used = p.run("Torque for wheel nuts")
        finally:
            p.retriever.search = real
        self.assertTrue(any(r.value == "204" for r in recs))
        self.assertIn("widened", used)


class TestNaturalLanguageQuestions(unittest.TestCase):
    """Full-sentence questions (not only keyword queries). Expected values were read by hand from the manual."""

    CASES = [
        ("What torque should be applied to the tie-rod end jam nut during front toe adjustment?", "jam nut", "103"),
        ("How tight should the front brake caliper guide pin bolts be?", "guide pin", "37"),
        ("What is the torque for the rear brake caliper guide pin bolts?", "guide pin", "33"),
        ("What is the wheel nut torque?", "Wheel nuts", "204"),
        ("What is the minimum thickness of the front brake disc?", "Front brake disc", "32"),
        ("What is the tightening torque of the halfshaft nut?", "Halfshaft nut", "40"),
        ("What torque should the stabilizer bar link nuts be tightened to?", "Stabilizer bar link nuts", "70"),
        ("Which part number is the shock absorber lower bolt?", "Shock absorber lower bolt", "W506545"),
        ("What is the maximum allowable driveshaft runout?", "Driveshaft", "0.50"),
        ("What torque do the differential housing cover bolts need?", "Differential housing cover bolts", "45"),
    ]

    def test_questions(self):
        for q, comp, value in self.CASES:
            with self.subTest(q=q):
                self.assertTrue(has(rows(q), comp, value), msg=f"{q} -> {rows(q)}")

    def test_jam_nut_component_name_and_section(self):
        rs = rows("Torque for the tie-rod end jam nut")
        self.assertTrue(has(rs, "Tie-rod end jam nut", "103", "Nm", "204-00"))

    def test_generic_nuts_step_keeps_full_part_name(self):
        # "Install the 2 new stabilizer bar link lower nuts. Tighten the nuts to 80 Nm"
        self.assertTrue(has(rows("Torque for stabilizer bar link nuts"), "Stabilizer bar link lower nuts", "80", "Nm"))


class TestPdfLayouts(unittest.TestCase):
    """PDF libraries differ in how they lay out tables; every layout must give the same specifications."""

    @staticmethod
    def _records(raw):
        from src.chunking import chunk_documents
        from src.extractor import table_records
        from src.pdf_extract import parse_manual
        return {(r.component.lower(), r.spec_type, r.value, r.unit, r.section_id)
                for c in chunk_documents(parse_manual(raw)) if c.is_table for r in table_records(c)}

    def test_reflow_torque_cells(self):
        from src.reflow import reflow
        cells = ["Description", "Nm", "lb-ft", "lb-in", "Wheel hub nut", "40", "30", "—",
                 "Integrated Wheel End (IWE) bolts", "12", "—", "106"]
        self.assertEqual(reflow(cells), ["Description Nm lb-ft lb-in", "Wheel hub nut 40 30 —",
                                         "Integrated Wheel End (IWE) bolts 12 — 106"])

    def test_reflow_spec_and_parts_cells(self):
        from src.reflow import reflow
        spec = ["Item", "Specification", "Front brake disc minimum thickness", "32 mm (1.259 in)"]
        self.assertEqual(reflow(spec), ["Item Specification", "Front brake disc minimum thickness 32 mm (1.259 in)"])
        parts = ["Item", "Part Number", "Description", "1", "W506545", "Shock absorber lower bolt",
                 "7", "18A149-B", "Shock absorber shield bolt (3 required)"]
        self.assertEqual(reflow(parts), ["Item Part Number Description", "1 W506545 Shock absorber lower bolt",
                                         "7 18A149-B Shock absorber shield bolt (3 required)"])

    def test_reflow_leaves_normal_text_alone(self):
        from src.reflow import reflow
        from src.pdf_extract import parse_manual
        lines = ["Description Nm lb-ft lb-in", "Wheel hub nut 40 30 —", "Tighten to 37 Nm (27 lb-ft)."]
        self.assertEqual(reflow(lines), lines)

    def test_flattened_manual_gives_the_same_specifications(self):
        sys.path.insert(0, str(ROOT / "tests"))
        from flatten import flatten
        raw = (config.DATA_DIR / "sample-service-manual.txt").read_text(encoding="utf-8")
        a, b = self._records(raw), self._records(flatten(raw))
        self.assertGreater(len(a), 300)
        self.assertEqual(a - b, set(), msg="specifications lost when cells arrive one per line")

    def test_rows_from_words(self):
        from src.pdf_extract import rows_from_words
        w = lambda x0, y, t: (x0, y, x0 + 30, y + 10, t, 0, 0, 0)       # noqa: E731
        words = [w(200, 101, "37"), w(10, 100, "Guide"), w(60, 100, "pin"), w(250, 100, "27"), w(10, 130, "Next"),
                 w(300, 100, "—")]
        self.assertEqual(rows_from_words(words), ["Guide pin 37 27 —", "Next"])

    def test_best_strategy_is_chosen_by_specification_yield(self):
        import tempfile
        from src import pdf_extract
        sys.path.insert(0, str(ROOT / "tests"))
        from flatten import flatten
        raw = (config.DATA_DIR / "sample-service-manual.txt").read_text(encoding="utf-8")
        broken = "\n".join(ln for ln in raw.split("\n") if "Nm" not in ln)      # a strategy that loses the torque tables
        old_strats, old_cache = pdf_extract.STRATEGIES, config.CACHE_DIR
        try:
            with tempfile.TemporaryDirectory() as d:
                config.CACHE_DIR = Path(d)
                pdf = Path(d) / "m.pdf"
                pdf.write_bytes(b"%PDF-fake")
                pdf_extract.STRATEGIES = [("bad", lambda p: broken), ("flat", lambda p: flatten(raw)),
                                          ("crash", lambda p: 1 / 0)]
                text, info = pdf_extract.read_raw_info(pdf)
                self.assertEqual(info["strategy"], "flat")
                self.assertIn("crash", info["errors"])
                self.assertTrue(pdf_extract.read_raw_info(pdf)[1]["cached"])     # second call uses the cache
        finally:
            pdf_extract.STRATEGIES, config.CACHE_DIR = old_strats, old_cache

    def test_clean_line_handles_tabs_and_zero_width_characters(self):
        from src.pdf_extract import _clean_line
        self.assertEqual(_clean_line("Wheel\thub\tnut\t40\t30\t\u2014"), "Wheel hub nut 40 30 \u2014")
        self.assertEqual(_clean_line("CXY\u200b80W90-1L"), "CXY80W90-1L")


class TestRegionAlternatives(unittest.TestCase):
    def test_front_and_rear_are_alternatives(self):
        qs = parse_query("What are the torque specifications for the front and rear brake caliper components?")
        self.assertEqual(qs.keywords, ["brake", "caliper"])
        self.assertEqual(sorted(qs.regions), ["front", "rear"])

    def test_front_and_rear_question_returns_both(self):
        rs = rows("What are the torque specifications for the front and rear brake caliper components?")
        self.assertTrue(has(rs, "guide pin", "37", "Nm", "206-03"))
        self.assertTrue(has(rs, "guide pin", "33", "Nm", "206-04"))
        self.assertTrue(has(rs, "anchor plate", "250", "Nm"))
        self.assertTrue(has(rs, "support bracket", "150", "Nm"))

    def test_minimum_front_brake_disc_thickness(self):
        self.assertTrue(has(rows("What is the minimum front brake disc thickness?"), "brake disc", "32", "mm"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
