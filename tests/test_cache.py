"""
Unit tests for SymptomCache.
Verifies sub-2-second retrieval for cached queries, normalization, and expiration.
"""

import time
import pytest
from modules.cache_service import SymptomCache


def test_cache_set_and_sub_two_second_retrieval(tmp_path):
    cache = SymptomCache(cache_dir=str(tmp_path), ttl_seconds=3600)
    symptoms = "Severe retrosternal chest pain radiating to left shoulder"
    dummy_data = {
        "diagnoses": [
            {"name": "Acute Coronary Syndrome", "likelihood": "High", "treatments": ["Aspirin", "Nitroglycerin"]}
        ]
    }

    # First access: Miss
    assert cache.get(symptoms) is None

    # Store in cache
    cache.set(symptoms, dummy_data)

    # Second access: Hit - must return within 2 seconds
    start_time = time.time()
    hit_result = cache.get(symptoms)
    elapsed = time.time() - start_time

    assert hit_result is not None
    data, timestamp = hit_result
    assert data["diagnoses"][0]["name"] == "Acute Coronary Syndrome"
    assert elapsed < 2.0, f"Cache retrieval took {elapsed}s, which exceeds 2s threshold!"


def test_cache_normalization(tmp_path):
    cache = SymptomCache(cache_dir=str(tmp_path), ttl_seconds=3600)
    query_a = "Severe headache with photophobia and nausea."
    query_b = "   severe  headache with photophobia and nausea   "
    dummy_data = {"diagnoses": [{"name": "Migraine"}]}

    cache.set(query_a, dummy_data)
    result = cache.get(query_b)

    assert result is not None
    data, _ = result
    assert data["diagnoses"][0]["name"] == "Migraine"


def test_cache_clear(tmp_path):
    cache = SymptomCache(cache_dir=str(tmp_path), ttl_seconds=3600)
    cache.set("Cough and fever", {"diagnoses": [{"name": "Bronchitis"}]})
    assert cache.get("Cough and fever") is not None

    cache.clear()
    assert cache.get("Cough and fever") is None
    stats = cache.get_stats()
    assert stats["total_cached_items"] == 0


def test_cache_ttl_expiration(tmp_path):
    # Set TTL to 1 second
    cache = SymptomCache(cache_dir=str(tmp_path), ttl_seconds=1)
    cache.set("Sore throat", {"diagnoses": [{"name": "Pharyngitis"}]})

    # Immediate access works
    assert cache.get("Sore throat") is not None

    # Wait for TTL to expire
    time.sleep(1.1)
    assert cache.get("Sore throat") is None
