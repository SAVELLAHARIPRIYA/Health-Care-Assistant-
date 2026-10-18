"""
HealthTrack AI - Comprehensive Test Runner using Python's built-in unittest
Covers:
1. Symptom Input Validation & Sanitization
2. Caching Layer Sub-2s SLA & Deterministic Normalization
3. Audit Logging Persistence
4. Clinical Analysis Differential Diagnosis (>= 3 guarantees)
5. Enterprise Resilience (Circuit Breaker, Exponential Backoff, Rate Limiting)
"""

import os
import sys
import time
import unittest

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from modules.validator import SymptomValidator
from modules.cache_service import SymptomCache
from modules.logger_service import AuditLogger
from modules.llm_service import ClinicalAnalysisService
from modules.resilience import (
    CircuitBreaker,
    CircuitState,
    retry_with_backoff,
    TokenBucketRateLimiter,
)


class TestSymptomValidator(unittest.TestCase):
    def test_empty_and_whitespace_input(self):
        res1 = SymptomValidator.validate("")
        self.assertFalse(res1.is_valid)
        self.assertIn("cannot be empty", res1.error_message.lower())

        res2 = SymptomValidator.validate("   ")
        self.assertFalse(res2.is_valid)

        res3 = SymptomValidator.validate(None)
        self.assertFalse(res3.is_valid)

    def test_too_short_input(self):
        res = SymptomValidator.validate("headache")
        self.assertFalse(res.is_valid)
        self.assertIn("too short", res.error_message.lower())

    def test_disallowed_special_characters(self):
        res_tag = SymptomValidator.validate("Patient has fever <script>alert(1)</script>")
        self.assertFalse(res_tag.is_valid)

        res_sym = SymptomValidator.validate("Patient has $$$ severe fever ### @@@ 102")
        self.assertFalse(res_sym.is_valid)
        self.assertIn("special symbols", res_sym.error_message.lower())

        res_num = SymptomValidator.validate("123456789012345")
        self.assertFalse(res_num.is_valid)

    def test_valid_clinical_descriptions(self):
        valid_samples = [
            "Patient presents with persistent dry cough, low-grade fever (38.2°C), and fatigue for 3 days.",
            "Severe throbbing left-sided headache accompanied by photophobia, phonophobia, and nausea.",
            "Right lower quadrant abdominal pain with tenderness at McBurney's point and low-grade fever.",
        ]
        for sample in valid_samples:
            res = SymptomValidator.validate(sample)
            self.assertTrue(res.is_valid, f"Validation unexpectedly failed for: {sample}")
            self.assertIsNone(res.error_message)


class TestSymptomCache(unittest.TestCase):
    def setUp(self):
        self.test_cache_dir = os.path.join(PROJECT_ROOT, "cache", "test_scratch")
        self.cache = SymptomCache(cache_dir=self.test_cache_dir, ttl_seconds=3600)

    def tearDown(self):
        self.cache.clear()
        if os.path.exists(self.test_cache_dir):
            try:
                import shutil
                shutil.rmtree(self.test_cache_dir)
            except Exception:
                pass

    def test_cache_hit_and_sub_two_second_sla(self):
        symptoms = "Severe retrosternal chest pain radiating to left shoulder"
        dummy_data = {
            "diagnoses": [
                {"name": "Acute Coronary Syndrome", "likelihood": "High", "treatments": ["Aspirin", "Nitroglycerin"]}
            ]
        }

        self.assertIsNone(self.cache.get(symptoms))
        self.cache.set(symptoms, dummy_data)

        start_t = time.time()
        hit = self.cache.get(symptoms)
        elapsed = time.time() - start_t

        self.assertIsNotNone(hit)
        data, _ = hit
        self.assertEqual(data["diagnoses"][0]["name"], "Acute Coronary Syndrome")
        self.assertLess(elapsed, 2.0, f"Cache retrieval took {elapsed}s, exceeding 2s SLA!")

    def test_cache_normalization(self):
        s1 = "Severe headache with photophobia and nausea."
        s2 = "   severe  headache with photophobia and nausea   "
        self.cache.set(s1, {"diagnoses": [{"name": "Migraine"}]})
        hit = self.cache.get(s2)
        self.assertIsNotNone(hit)
        self.assertEqual(hit[0]["diagnoses"][0]["name"], "Migraine")


class TestAuditLogger(unittest.TestCase):
    def setUp(self):
        self.test_log_path = os.path.join(PROJECT_ROOT, "logs", "test_audit.txt")
        self.logger = AuditLogger(log_file_path=self.test_log_path)

    def tearDown(self):
        if os.path.exists(self.test_log_path):
            try:
                os.remove(self.test_log_path)
            except Exception:
                pass

    def test_logs_input_and_output_to_text_file(self):
        symptoms = "Sudden onset of severe crushing chest pain radiating to left jaw."
        output = {
            "diagnoses": [
                {"name": "Acute Myocardial Infarction", "likelihood": "High", "treatments": ["Aspirin"]}
            ]
        }

        query_id = self.logger.log_assessment(
            symptoms=symptoms,
            output_data=output,
            source="UNIT_TEST",
            response_time_sec=0.05,
            status="SUCCESS",
        )

        self.assertTrue(os.path.exists(self.test_log_path))
        with open(self.test_log_path, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn(symptoms, content)
        self.assertIn("Acute Myocardial Infarction", content)
        self.assertIn("Aspirin", content)
        self.assertIn(query_id, content)


class TestClinicalAnalysisService(unittest.TestCase):
    def setUp(self):
        self.service = ClinicalAnalysisService(api_key=None)

    def test_guarantees_minimum_three_diagnoses(self):
        symptoms = "Patient presents with persistent cough, fever of 38.5C, and pleuritic chest pain."
        result = self.service.analyze_symptoms(symptoms)

        self.assertIn("diagnoses", result)
        diagnoses = result["diagnoses"]

        self.assertGreaterEqual(
            len(diagnoses), 3, f"Expected >= 3 diagnoses, got {len(diagnoses)}"
        )

        for diag in diagnoses:
            self.assertTrue(diag.get("name"))
            self.assertIn(diag.get("likelihood"), ["High", "Moderate", "Low"])
            self.assertIsInstance(diag.get("treatments"), list)
            self.assertGreaterEqual(len(diag.get("treatments")), 1)


class TestResilienceAndCircuitBreaker(unittest.TestCase):
    def test_circuit_breaker_trips_to_open_after_threshold(self):
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout_sec=1.0)
        self.assertEqual(cb.state, CircuitState.CLOSED)
        self.assertTrue(cb.can_execute())

        cb.record_failure(Exception("Error 1"))
        self.assertEqual(cb.state, CircuitState.CLOSED)

        cb.record_failure(Exception("Error 2"))
        self.assertEqual(cb.state, CircuitState.CLOSED)

        # 3rd failure trips the circuit
        cb.record_failure(Exception("Error 3"))
        self.assertEqual(cb.state, CircuitState.OPEN)
        self.assertFalse(cb.can_execute())

        # Recovers to HALF_OPEN after timeout
        time.sleep(1.05)
        self.assertEqual(cb.state, CircuitState.HALF_OPEN)
        self.assertTrue(cb.can_execute())

        # Successful test request closes circuit
        cb.record_success()
        self.assertEqual(cb.state, CircuitState.CLOSED)

    def test_exponential_backoff_decorator_retries(self):
        call_count = 0

        @retry_with_backoff(max_retries=3, base_delay=0.01, max_delay=0.05, jitter=False)
        def flaky_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Transient error")
            return "SUCCESS"

        result = flaky_function()
        self.assertEqual(result, "SUCCESS")
        self.assertEqual(call_count, 3)

    def test_token_bucket_rate_limiter(self):
        limiter = TokenBucketRateLimiter(max_tokens=2, refill_period_sec=60.0)

        # First 2 tokens acquired immediately
        allowed1, _ = limiter.acquire()
        allowed2, _ = limiter.acquire()
        self.assertTrue(allowed1)
        self.assertTrue(allowed2)

        # 3rd token blocked (rate limit exceeded)
        allowed3, wait_time = limiter.acquire()
        self.assertFalse(allowed3)
        self.assertGreater(wait_time, 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
