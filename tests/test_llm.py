"""
Unit tests for ClinicalAnalysisService.
Verifies differential diagnosis parsing and the guarantee of at least 3 diagnoses.
"""

import pytest
from modules.llm_service import ClinicalAnalysisService


def test_clinical_service_guarantees_minimum_three_diagnoses():
    service = ClinicalAnalysisService(api_key=None)  # Uses demonstration mock engine

    test_symptoms = "Patient presents with persistent cough, fever of 38.5C, and pleuritic chest pain."
    result = service.analyze_symptoms(test_symptoms)

    assert "diagnoses" in result
    diagnoses = result["diagnoses"]

    # MUST REQUIREMENT: at least 3 potential diagnoses
    assert len(diagnoses) >= 3, f"Expected at least 3 diagnoses, got {len(diagnoses)}"

    # Check structure of each diagnosis
    for diag in diagnoses:
        assert "name" in diag and diag["name"]
        assert "likelihood" in diag
        assert "treatments" in diag
        assert isinstance(diag["treatments"], list)
        assert len(diag["treatments"]) >= 1, f"Diagnosis {diag['name']} missing treatments"


def test_clinical_service_various_presentations():
    service = ClinicalAnalysisService(api_key=None)

    # Test Abdominal pain presentation
    res_abd = service.analyze_symptoms("Severe right lower quadrant abdominal pain with rebound tenderness")
    assert len(res_abd["diagnoses"]) >= 3
    assert any("appendicitis" in d["name"].lower() for d in res_abd["diagnoses"])

    # Test Headache presentation
    res_head = service.analyze_symptoms("Severe unilateral throbbing headache with photophobia and aura")
    assert len(res_head["diagnoses"]) >= 3
    assert any("migraine" in d["name"].lower() for d in res_head["diagnoses"])

    # Test Chest Pain presentation
    res_chest = service.analyze_symptoms("Substernal pressure radiating to left arm and jaw with diaphoresis")
    assert len(res_chest["diagnoses"]) >= 3
    assert any("coronary" in d["name"].lower() for d in res_chest["diagnoses"])


def test_ensure_minimum_diagnoses_fallback_supplementation():
    service = ClinicalAnalysisService(api_key=None)
    sparse_data = {
        "diagnoses": [
            {"name": "Diagnosis A", "likelihood": "High", "treatments": ["Tx A"]}
        ]
    }
    supplemented = service._ensure_minimum_diagnoses(sparse_data, "Generic symptoms")
    assert len(supplemented["diagnoses"]) >= 3
