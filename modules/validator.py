"""
HealthTrack AI - Symptom Input Validation Module

Validates symptom inputs to ensure they meet clinical system integrity,
format requirements, and security guardrails before reaching the LLM.
"""

import re
from typing import NamedTuple, Optional


class ValidationResult(NamedTuple):
    is_valid: bool
    error_message: Optional[str]
    sanitized_text: str


class SymptomValidator:
    """Validates patient symptom input according to medical assessment rules."""

    MIN_LENGTH = 10
    MAX_LENGTH = 2000

    # Allowed characters: letters, numbers, spaces, and safe medical punctuation
    # (.,;:'"-()/%\n\r+)
    ALLOWED_PATTERN = re.compile(r"^[a-zA-Z0-9\s.,;:'\"\-()/%\n\r+°]+$")

    # Forbidden dangerous patterns (scripts, tags, code injection attempts)
    DISALLOWED_PATTERNS = [
        (re.compile(r"<[^>]*>"), "HTML/XML tags are not permitted in clinical symptom descriptions."),
        (re.compile(r"[{}[\]\\^`~*|<>]"), "Special code or scripting symbols ({}[ ]\\^`~*|<>) are not allowed."),
        (re.compile(r"(drop\s+table|select\s+\*|union\s+select|insert\s+into)", re.IGNORECASE), "Invalid input pattern detected."),
        (re.compile(r"(\$|#|@|_|=)"), "Special symbols ($, #, @, _, =) are not allowed in the symptom description."),
    ]

    @classmethod
    def validate(cls, raw_input: Optional[str]) -> ValidationResult:
        """
        Validate patient symptom input string.

        Args:
            raw_input: The raw symptom description entered by the user.

        Returns:
            ValidationResult with boolean `is_valid`, `error_message`, and `sanitized_text`.
        """
        if raw_input is None or not raw_input.strip():
            return ValidationResult(
                is_valid=False,
                error_message="Symptom input cannot be empty. Please enter the patient's symptoms.",
                sanitized_text="",
            )

        trimmed = raw_input.strip()

        # Check minimum length
        if len(trimmed) < cls.MIN_LENGTH:
            return ValidationResult(
                is_valid=False,
                error_message=(
                    f"Symptom description is too short ({len(trimmed)} chars). "
                    f"Please provide at least {cls.MIN_LENGTH} characters for an accurate clinical assessment."
                ),
                sanitized_text=trimmed,
            )

        # Check maximum length
        if len(trimmed) > cls.MAX_LENGTH:
            return ValidationResult(
                is_valid=False,
                error_message=(
                    f"Symptom description exceeds maximum limit ({len(trimmed)}/{cls.MAX_LENGTH} chars). "
                    "Please condense the description."
                ),
                sanitized_text=trimmed[: cls.MAX_LENGTH],
            )

        # Must contain at least one word character / alphabetic character
        if not re.search(r"[a-zA-Z]", trimmed):
            return ValidationResult(
                is_valid=False,
                error_message="Symptom input must contain descriptive medical text, not just numbers or symbols.",
                sanitized_text=trimmed,
            )

        # Check for disallowed injection/code patterns
        for pattern, err_msg in cls.DISALLOWED_PATTERNS:
            if pattern.search(trimmed):
                return ValidationResult(
                    is_valid=False,
                    error_message=err_msg,
                    sanitized_text=trimmed,
                )

        # Verify allowed character set
        if not cls.ALLOWED_PATTERN.match(trimmed):
            invalid_chars = set(c for c in trimmed if not cls.ALLOWED_PATTERN.match(c))
            return ValidationResult(
                is_valid=False,
                error_message=(
                    f"Input contains disallowed special characters: {' '.join(invalid_chars)}. "
                    "Please use standard text and common punctuation (commas, periods, hyphens)."
                ),
                sanitized_text=trimmed,
            )

        # Repetitive character spam detection (e.g., 'aaaaaa' or 'zzzzzzzz')
        if re.search(r"(.)\1{7,}", trimmed):
            return ValidationResult(
                is_valid=False,
                error_message="Input contains excessive repeated characters. Please enter genuine patient symptoms.",
                sanitized_text=trimmed,
            )

        # Clean multiple spaces while preserving single spaces and linebreaks
        sanitized = re.sub(r"[ \t]+", " ", trimmed)

        return ValidationResult(
            is_valid=True,
            error_message=None,
            sanitized_text=sanitized,
        )
