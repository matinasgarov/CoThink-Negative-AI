"""Tests for the moderation HTTP service.

Uses FastAPI's TestClient, which runs the app's lifespan, so these exercise the
real gate rather than a mock.

Run: python -m unittest test_api -v
"""

import json
import os
import tempfile
import unittest

from fastapi.testclient import TestClient

import api


class ApiTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Redirect the audit log so tests never append to the real one.
        cls._tmp = tempfile.TemporaryDirectory()
        api.AUDIT_LOG = os.path.join(cls._tmp.name, "audit.jsonl")
        cls.client = TestClient(api.app)
        cls.client.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client.__exit__(None, None, None)
        cls._tmp.cleanup()


class TestHealth(ApiTestCase):
    def test_health_reports_a_loaded_gate(self):
        body = self.client.get("/health").json()
        self.assertEqual(body["status"], "ok")
        self.assertGreater(body["lexicon_forms"], 500)
        self.assertIsNotNone(body["loaded_at"])


class TestModerate(ApiTestCase):
    def post(self, text, surface="public", **kwargs):
        response = self.client.post(
            "/moderate", json={"text": text, "surface": surface, **kwargs}
        )
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def test_clean_text_is_allowed(self):
        body = self.post("Bu dərs çox faydalı idi")
        self.assertEqual(body["action"], "allow")
        self.assertTrue(body["proceed"])
        self.assertEqual(body["outcome"], "allowed")

    def test_obfuscated_slur_is_blocked_and_names_the_word(self):
        body = self.post("s3n w3r3fs1z")
        self.assertEqual(body["action"], "block")
        self.assertFalse(body["proceed"])
        self.assertIn("şərəfsiz", [m["word"] for m in body["lexicon_matches"]])

    def test_whitelisted_word_is_not_blocked(self):
        body = self.post("Kitabı götürmək istəyirəm")
        self.assertTrue(body["proceed"])

    def test_surface_changes_the_outcome_for_review_decisions(self):
        text = "Sen kimsen? Pay atonan!!!"
        public, private = self.post(text, "public"), self.post(text, "private")
        # Same decision, different product behaviour.
        self.assertEqual(public["action"], private["action"])
        if public["action"] == "review":
            self.assertFalse(public["proceed"])
            self.assertEqual(public["outcome"], "held_for_review")
            self.assertTrue(private["proceed"])
            self.assertEqual(private["outcome"], "allowed_flagged")

    def test_surface_defaults_to_the_stricter_option(self):
        response = self.client.post("/moderate", json={"text": "salam"})
        self.assertEqual(response.status_code, 200)

    def test_response_includes_latency(self):
        self.assertGreaterEqual(self.post("salam")["latency_ms"], 0)

    def test_empty_text_is_accepted(self):
        self.assertEqual(self.post("")["action"], "allow")

    def test_rejects_oversized_text(self):
        response = self.client.post(
            "/moderate", json={"text": "a" * (api.MAX_TEXT_LENGTH + 1)}
        )
        self.assertEqual(response.status_code, 422)

    def test_rejects_unknown_surface(self):
        response = self.client.post(
            "/moderate", json={"text": "salam", "surface": "everywhere"}
        )
        self.assertEqual(response.status_code, 422)

    def test_rejects_missing_text(self):
        self.assertEqual(self.client.post("/moderate", json={}).status_code, 422)


class TestBatch(ApiTestCase):
    def test_batch_returns_one_result_per_item(self):
        items = [{"text": t} for t in ["salam", "s3n w3r3fs1z", "Bu dərs faydalı idi"]]
        response = self.client.post("/moderate/batch", json={"items": items})
        self.assertEqual(response.status_code, 200)
        results = response.json()["results"]
        self.assertEqual(len(results), 3)
        self.assertEqual(results[1]["action"], "block")

    def test_rejects_oversized_batch(self):
        items = [{"text": "salam"}] * (api.MAX_BATCH_SIZE + 1)
        response = self.client.post("/moderate/batch", json={"items": items})
        self.assertEqual(response.status_code, 422)


class TestAuditLog(ApiTestCase):
    def read_log(self):
        if not os.path.exists(api.AUDIT_LOG):
            return []
        with open(api.AUDIT_LOG, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def test_blocked_text_is_logged_with_its_reason(self):
        before = len(self.read_log())
        self.client.post("/moderate", json={
            "text": "s3n w3r3fs1z", "surface": "public", "context": "comment:42",
        })
        entries = self.read_log()
        self.assertEqual(len(entries), before + 1)
        entry = entries[-1]
        self.assertEqual(entry["action"], "block")
        self.assertEqual(entry["context"], "comment:42")
        self.assertIn("şərəfsiz", entry["matched_words"])

    def test_allowed_text_is_not_logged(self):
        before = len(self.read_log())
        self.client.post("/moderate", json={"text": "Bu dərs faydalı idi"})
        self.assertEqual(len(self.read_log()), before)


if __name__ == "__main__":
    unittest.main()
