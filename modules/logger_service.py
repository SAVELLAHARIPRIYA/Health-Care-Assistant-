"""
HealthTrack AI - Audit Logging Service

Records healthcare professional inputs and LLM outputs to a persistent text log file
for clinical compliance, retrospective audit, and safety analysis.
"""

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional
import uuid


class AuditLogger:
    """Manages clinical assessment audit trails."""

    DEFAULT_LOG_PATH = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "logs",
        "audit_log.txt",
    )

    def __init__(self, log_file_path: Optional[str] = None):
        self.log_file_path = log_file_path or self.DEFAULT_LOG_PATH
        self._ensure_log_directory()

    def _ensure_log_directory(self) -> None:
        """Ensure the parent directory for the log file exists."""
        log_dir = os.path.dirname(self.log_file_path)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

    def log_assessment(
        self,
        symptoms: str,
        output_data: Dict[str, Any],
        source: str = "GEMINI_API",
        response_time_sec: float = 0.0,
        patient_context: Optional[Dict[str, Any]] = None,
        status: str = "SUCCESS",
        error_message: Optional[str] = None,
    ) -> str:
        """
        Record a single patient symptom query and model output to the audit log text file.

        Args:
            symptoms: The input symptom text.
            output_data: Structured diagnosis and treatment suggestions dictionary.
            source: 'CACHE' or 'GEMINI_API' or 'MOCK_ENGINE'.
            response_time_sec: Time taken in seconds.
            patient_context: Optional context like age, sex, symptom duration.
            status: 'SUCCESS' or 'ERROR'.
            error_message: Optional error message if status is 'ERROR'.

        Returns:
            The unique query_id generated for this entry.
        """
        query_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Format human-readable text entry for audit_log.txt
        diagnoses_summary = []
        if output_data and "diagnoses" in output_data:
            for i, diag in enumerate(output_data.get("diagnoses", []), 1):
                name = diag.get("name", "Unknown")
                likelihood = diag.get("likelihood", "N/A")
                treatments = ", ".join(diag.get("treatments", []))
                diagnoses_summary.append(
                    f"    {i}. {name} [Likelihood: {likelihood}] -> Treatments: {treatments}"
                )
        diagnoses_str = (
            "\n".join(diagnoses_summary)
            if diagnoses_summary
            else "    (No structured diagnoses generated)"
        )

        patient_meta_str = ""
        if patient_context:
            patient_meta_str = f"Patient Context: {json.dumps(patient_context)}\n"

        log_entry = (
            f"================================================================================\n"
            f"[AUDIT LOG ENTRY] ID: {query_id} | Timestamp: {timestamp}\n"
            f"Source: {source} | Status: {status} | Latency: {response_time_sec:.3f}s\n"
            f"{patient_meta_str}"
            f"INPUT SYMPTOMS:\n"
            f"  \"{symptoms}\"\n\n"
            f"MODEL OUTPUT DIAGNOSES & TREATMENTS:\n"
            f"{diagnoses_str}\n"
        )

        if error_message:
            log_entry += f"\nERROR DETAILS: {error_message}\n"

        log_entry += f"================================================================================\n\n"

        try:
            with open(self.log_file_path, "a", encoding="utf-8") as f:
                f.write(log_entry)
        except Exception as e:
            print(f"[AuditLogger Error] Failed to write to {self.log_file_path}: {e}")

        return query_id

    def read_recent_logs(self, max_entries: int = 20) -> List[str]:
        """Read the most recent entries from the log file for display in the audit UI."""
        if not os.path.exists(self.log_file_path):
            return []

        try:
            with open(self.log_file_path, "r", encoding="utf-8") as f:
                content = f.read()

            entries = [
                entry.strip()
                for entry in content.split("================================================================================")
                if entry.strip()
            ]
            return entries[-max_entries:]
        except Exception as e:
            return [f"Error reading audit log: {e}"]
