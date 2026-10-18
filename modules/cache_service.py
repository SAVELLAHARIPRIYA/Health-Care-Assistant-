"""
HealthTrack AI - Symptom Caching Layer

Provides in-memory and persistent caching for repeated symptom queries to reduce latency,
conserve LLM token costs, and guarantee sub-2-second retrieval times for recurrent presentations.
"""

import hashlib
import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple


class SymptomCache:
    """Thread-safe caching layer with key normalization and disk persistence."""

    DEFAULT_CACHE_DIR = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "cache",
    )
    DEFAULT_CACHE_FILE = "symptom_cache_store.json"

    def __init__(self, cache_dir: Optional[str] = None, ttl_seconds: int = 86400):
        self.cache_dir = cache_dir or self.DEFAULT_CACHE_DIR
        self.cache_file = os.path.join(self.cache_dir, self.DEFAULT_CACHE_FILE)
        self.ttl_seconds = ttl_seconds
        self._memory_cache: Dict[str, Dict[str, Any]] = {}
        self.hits = 0
        self.misses = 0

        self._ensure_cache_directory()
        self._load_from_disk()

    def _ensure_cache_directory(self) -> None:
        """Ensure cache storage folder exists."""
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)

    def _normalize_symptoms(self, symptoms: str) -> str:
        """
        Normalize symptom text: lowercase, collapse multiple spaces, strip edge punctuation.
        Ensures consistent cache keying across slight formatting differences.
        """
        normalized = symptoms.strip().lower()
        normalized = re.sub(r"\s+", " ", normalized)
        normalized = re.sub(r"^[^\w]+|[^\w]+$", "", normalized)
        return normalized

    def _generate_cache_key(
        self, symptoms: str, patient_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate a deterministic hash key from normalized symptoms and context."""
        norm_symptoms = self._normalize_symptoms(symptoms)
        context_str = json.dumps(patient_context or {}, sort_keys=True)
        raw_key = f"{norm_symptoms}::{context_str}"
        return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()

    def get(
        self, symptoms: str, patient_context: Optional[Dict[str, Any]] = None
    ) -> Optional[Tuple[Dict[str, Any], float]]:
        """
        Retrieve cached diagnosis output if present and not expired.

        Returns:
            Tuple of (response_data, cached_timestamp) or None if miss.
        """
        key = self._generate_cache_key(symptoms, patient_context)
        entry = self._memory_cache.get(key)

        if not entry:
            self.misses += 1
            return None

        # Check TTL expiration
        timestamp = entry.get("timestamp", 0)
        if time.time() - timestamp > self.ttl_seconds:
            # Expired
            del self._memory_cache[key]
            self.misses += 1
            self._save_to_disk()
            return None

        self.hits += 1
        entry["hit_count"] = entry.get("hit_count", 0) + 1
        return entry.get("data"), timestamp

    def set(
        self,
        symptoms: str,
        data: Dict[str, Any],
        patient_context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Store analysis results in both memory and persistent disk cache."""
        key = self._generate_cache_key(symptoms, patient_context)
        entry = {
            "symptoms_preview": symptoms[:100],
            "data": data,
            "timestamp": time.time(),
            "hit_count": 0,
        }
        self._memory_cache[key] = entry
        self._save_to_disk()

    def clear(self) -> None:
        """Clear all in-memory and disk cache."""
        self._memory_cache.clear()
        self.hits = 0
        self.misses = 0
        if os.path.exists(self.cache_file):
            try:
                os.remove(self.cache_file)
            except Exception:
                pass

    def get_stats(self) -> Dict[str, Any]:
        """Return cache health and usage statistics."""
        total_queries = self.hits + self.misses
        hit_rate = (self.hits / total_queries * 100) if total_queries > 0 else 0.0
        return {
            "total_cached_items": len(self._memory_cache),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate_pct": round(hit_rate, 1),
        }

    def list_entries(self) -> List[Dict[str, Any]]:
        """List summary of cached symptom items."""
        summaries = []
        for key, entry in self._memory_cache.items():
            summaries.append({
                "key": key[:8],
                "symptoms": entry.get("symptoms_preview", ""),
                "hits": entry.get("hit_count", 0),
                "timestamp": time.strftime(
                    "%Y-%m-%d %H:%M:%S", time.localtime(entry.get("timestamp", 0))
                ),
            })
        return summaries

    def _load_from_disk(self) -> None:
        """Load persistent cache from disk upon initialization."""
        if not os.path.exists(self.cache_file):
            return

        try:
            with open(self.cache_file, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                now = time.time()
                self._memory_cache = {
                    k: v
                    for k, v in loaded.items()
                    if now - v.get("timestamp", 0) <= self.ttl_seconds
                }
        except Exception as e:
            print(f"[Cache Load Warning] Could not load cache file: {e}")
            self._memory_cache = {}

    def _save_to_disk(self) -> None:
        """Save current memory cache state to disk JSON file."""
        try:
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self._memory_cache, f, indent=2)
        except Exception as e:
            print(f"[Cache Save Warning] Could not persist cache: {e}")
