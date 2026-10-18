"""
Unit tests for SymptomValidator.
Verifies empty checks, length limits, special character prevention, and friendly error messages.
"""

import pytest
from modules.validator import SymptomValidator


def test_empty_and_whitespace_input():
    res1 = SymptomValidator.validate("")
    assert not res1.is_valid
    assert "cannot be empty" in res1.error_message.lower()

    res2 = SymptomValidator.validate("    ")
    assert not res2.is_valid
    assert "cannot be empty" in res2.error_message.lower()

    res3 = SymptomValidator.validate(None)
    assert not res3.is_valid


def test_too_short_input():
    res = SymptomValidator.validate("headache")  # 8 chars < 10
    assert not res.is_valid
    assert "too short" in res.error_message.lower()


def test_disallowed_special_characters():
    # HTML tag injection
    res1 = SymptomValidator.validate("Patient has severe headache <script>alert(1)</script>")
    assert not res1.is_valid
    assert "html" in res1.error_message.lower() or "tags" in res1.error_message.lower()

    # Disallowed special symbols like $, #, @
    res2 = SymptomValidator.validate("Patient symptoms are $$$ fever ### @@@ 102 degrees")
    assert not res2.is_valid
    assert "special symbols" in res2.error_message.lower()

    # Pure numbers or symbols
    res3 = SymptomValidator.validate("123456789012345")
    assert not res3.is_valid
    assert "must contain descriptive medical text" in res3.error_message.lower()


def test_repeated_spam_input():
    res = SymptomValidator.validate("Patient has aaaaaaaaaaa headache")
    assert not res.is_valid
    assert "repeated characters" in res.error_message.lower()


def test_valid_clinical_symptom_descriptions():
    valid_inputs = [
        "Patient presents with persistent dry cough, low-grade fever (38.2°C), and fatigue for 3 days.",
        "Severe throbbing left-sided headache accompanied by photophobia, phonophobia, and nausea.",
        "Right lower quadrant abdominal pain with tenderness at McBurney's point and low-grade fever.",
    ]
    for text in valid_inputs:
        res = SymptomValidator.validate(text)
        assert res.is_valid, f"Failed for valid text: {text}. Error: {res.error_message}"
        assert res.error_message is None
        assert len(res.sanitized_text) >= 10
