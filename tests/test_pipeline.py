"""End-to-end and unit tests. Run with `python tests/test_pipeline.py` or `pytest -q` (offline, no API key)."""
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from src.extractor import Extractor, parse_llm_json, public, value_in_text  # noqa: E402
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
        self.assertTrue(has(rs, "guide pin", "37", "Nm", "206-03"))
        self.assertTrue(has(rs, "guide pin", "33", "Nm", "206-04"))
        self.assertTrue(has(rs, "anchor plate", "250", "Nm"))
        self.assertTrue(has(rs, "support bracket", "150", "Nm"))

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
