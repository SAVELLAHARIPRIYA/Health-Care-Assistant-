"""
Unit tests for AuditLogger.
Verifies that user inputs and model outputs are recorded in a text log file.
"""

import os
import pytest
from modules.logger_service import AuditLogger


def test_audit_logger_records_input_and_output(tmp_path):
    log_file = tmp_path / "test_audit.txt"
    logger = AuditLogger(log_file_path=str(log_file))

    symptoms = "Sudden onset of severe crushing chest pain radiating to left jaw."
    output_data = {
        "diagnoses": [
            {
                "name": "Acute Myocardial Infarction",
                "likelihood": "High",
                "treatments": ["Chewable Aspirin", "Sublingual Nitroglycerin"],
            },
            {
                "name": "Gastroesophageal Reflux",
                "likelihood": "Low",
                "treatments": ["Antacids"],
            },
        ]
    }

    query_id = logger.log_assessment(
        symptoms=symptoms,
        output_data=output_data,
        source="TEST_SUITE",
        response_time_sec=0.123,
        status="SUCCESS",
    )

    assert os.path.exists(log_file)

    with open(log_file, "r", encoding="utf-8") as f:
        content = f.read()

    # Verify input symptoms recorded
    assert symptoms in content
    # Verify diagnoses recorded
    assert "Acute Myocardial Infarction" in content
    assert "Chewable Aspirin" in content
    # Verify query id and status
    assert query_id in content
    assert "SUCCESS" in content


def test_recent_logs_reader(tmp_path):
    log_file = tmp_path / "test_audit.txt"
    logger = AuditLogger(log_file_path=str(log_file))

    for i in range(3):
        logger.log_assessment(
            symptoms=f"Symptom test presentation #{i}",
            output_data={"diagnoses": [{"name": f"Condition #{i}", "treatments": ["Rest"]}]},
            status="SUCCESS",
        )

    entries = logger.read_recent_logs(max_entries=10)
    assert len(entries) == 3
    assert "Condition #2" in entries[-1]
