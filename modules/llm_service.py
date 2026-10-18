"""
HealthTrack AI - Gemini LLM Clinical Analysis Service with Enterprise Resilience

Integrates Google Gemini to analyze patient symptoms, returning structured differential
diagnoses and treatment suggestions with clinical rigor and edge-case handling.
Includes Circuit Breaker, Exponential Backoff with Jitter, and Graceful Degradation.
"""

import json
import os
import re
import warnings
from typing import Any, Dict, List, Optional

from modules.resilience import (
    CircuitBreaker,
    CircuitBreakerOpenException,
    CircuitState,
    TokenBucketRateLimiter,
    retry_with_backoff,
)

# Attempt modern google.genai SDK first
GENAI_SDK = None

try:
    from google import genai
    from google.genai import types
    GENAI_SDK = "genai"
except ImportError:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import google.generativeai as legacy_genai
        GENAI_SDK = "legacy"
    except ImportError:
        GENAI_SDK = None


# Shared resilience singletons across requests
_SHARED_CIRCUIT_BREAKER = CircuitBreaker(failure_threshold=3, recovery_timeout_sec=30.0)
_SHARED_RATE_LIMITER = TokenBucketRateLimiter(max_tokens=15, refill_period_sec=60.0)


class ClinicalAnalysisService:
    """Service to handle symptom analysis using Gemini LLM with circuit-breaker protection and fallback."""

    DEFAULT_MODEL = "gemini-3.8-flash"

    SYSTEM_PROMPT = """You are HealthTrack AI, an advanced Clinical Decision Support System designed exclusively for healthcare professionals (physicians, nurses, and triage clinicians).

Your task is to analyze patient symptoms and provide an evidence-based differential diagnosis with corresponding actionable treatment suggestions.

MANDATORY CLINICAL REQUIREMENTS:
1. You MUST provide AT LEAST THREE (>= 3) potential diagnoses, ranked from highest to lowest clinical likelihood.
2. For EACH diagnosis, you MUST provide:
   - "name": Standard medical diagnosis name.
   - "likelihood": Categorized as "High", "Moderate", or "Low".
   - "rationale": Clear medical explanation correlating the reported symptoms with this disease pathology.
   - "treatments": A detailed list of at least 2-4 concrete treatment options (first-line medications, supportive therapy, clinical interventions, specialist consults).
   - "urgency": "Emergency", "Urgent", or "Routine".
   - "recommended_tests": List of recommended diagnostic workups (e.g., Labs, Imaging, Vitals).
3. Provide an array of "immediate_recommendations" for the attending clinician.
4. Provide an array of "red_flags" (alarm symptoms indicating life-threatening deterioration).
5. Always include a clinical disclaimer.

OUTPUT FORMAT:
You MUST output ONLY a valid JSON object. Do not include markdown preamble or conversational text. Follow this exact JSON schema:
{
  "diagnoses": [
    {
      "name": "Diagnosis Name",
      "likelihood": "High",
      "rationale": "Medical reasoning...",
      "treatments": ["Treatment option 1", "Treatment option 2", "Treatment option 3"],
      "urgency": "Urgent",
      "recommended_tests": ["Diagnostic test 1", "Diagnostic test 2"]
    },
    {
      "name": "Second Diagnosis",
      "likelihood": "Moderate",
      "rationale": "Medical reasoning...",
      "treatments": ["Treatment option 1", "Treatment option 2"],
      "urgency": "Routine",
      "recommended_tests": ["Test 1"]
    },
    {
      "name": "Third Diagnosis",
      "likelihood": "Low",
      "rationale": "Medical reasoning...",
      "treatments": ["Treatment option 1", "Treatment option 2"],
      "urgency": "Routine",
      "recommended_tests": ["Test 1"]
    }
  ],
  "immediate_recommendations": [
    "Immediate clinical action 1",
    "Monitoring recommendation"
  ],
  "red_flags": [
    "Alarm symptom requiring rapid emergency response"
  ],
  "disclaimer": "HealthTrack AI is an assistive decision-support tool for licensed healthcare professionals and does not replace autonomous clinical examination."
}
"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        circuit_breaker: Optional[CircuitBreaker] = None,
        rate_limiter: Optional[TokenBucketRateLimiter] = None,
    ):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self.model_name = model_name or os.getenv("GEMINI_MODEL", self.DEFAULT_MODEL)
        self.circuit_breaker = circuit_breaker or _SHARED_CIRCUIT_BREAKER
        self.rate_limiter = rate_limiter or _SHARED_RATE_LIMITER
        self.client = None
        self._configured = False

        if self.api_key:
            if GENAI_SDK == "genai":
                try:
                    self.client = genai.Client(api_key=self.api_key)
                    self._configured = True
                except Exception as e:
                    print(f"[google.genai Init Error]: {e}")
            elif GENAI_SDK == "legacy":
                try:
                    legacy_genai.configure(api_key=self.api_key)
                    self._configured = True
                except Exception as e:
                    print(f"[legacy_genai Init Error]: {e}")

    def is_live_configured(self) -> bool:
        """Check if live Gemini API is ready to use."""
        return bool(self._configured and self.api_key and GENAI_SDK is not None)

    def get_resilience_status(self) -> Dict[str, Any]:
        """Return live health status of circuit breaker and resilience components."""
        return self.circuit_breaker.get_status()

    def reset_circuit_breaker(self) -> None:
        """Manual reset of circuit breaker."""
        self.circuit_breaker.reset()

    def analyze_symptoms(
        self,
        symptoms: str,
        patient_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Analyze patient symptoms with full resilience lifecycle:
        1. Circuit Breaker Check (fail-fast if remote API is degraded)
        2. Client-side Rate Limit check
        3. Exponential Backoff + Jitter retries on transient errors
        4. Graceful Fallback if all retries fail
        """
        # Step 1: Circuit Breaker Inspection
        if not self.circuit_breaker.can_execute():
            # Circuit is OPEN! Fail fast immediately to prevent long clinical wait times
            cb_status = self.circuit_breaker.get_status()
            fallback = self._generate_fallback_diagnosis(symptoms, patient_context)
            fallback["_notice"] = (
                f"🛡️ Circuit Breaker TRIPPED to OPEN ({cb_status['consecutive_failures']} consecutive errors). "
                f"Fail-fast engaged: served immediately via Clinical Fallback Engine. "
                f"Cooldown remaining: {cb_status['remaining_cooldown_sec']}s."
            )
            fallback["_source"] = "CIRCUIT_BREAKER_FALLBACK"
            return fallback

        # If live API is configured, attempt call through resilience pipeline
        if self.is_live_configured():
            # Step 2: Rate limit check (Free-tier RPM guard)
            allowed, wait_sec = self.rate_limiter.acquire()
            if not allowed:
                fallback = self._generate_fallback_diagnosis(symptoms, patient_context)
                fallback["_notice"] = f"⏳ Client-side rate limit reached. Throttled for {wait_sec:.1f}s. Served via Clinical Fallback Engine."
                fallback["_source"] = "RATE_LIMIT_FALLBACK"
                return fallback

            # Step 3: Call with retries & circuit breaker accounting
            try:
                result = self._execute_with_retries(symptoms, patient_context)
                self.circuit_breaker.record_success()
                return result
            except Exception as e:
                # Record failure in circuit breaker
                self.circuit_breaker.record_failure(e)
                fallback = self._generate_fallback_diagnosis(symptoms, patient_context)
                fallback["_notice"] = f"Live Gemini API failed after retries ({str(e)}). Circuit failure counter incremented."
                fallback["_source"] = "FALLBACK_MOCK"
                return fallback

        # If API key not configured, use intelligent clinical mock engine
        mock_result = self._generate_fallback_diagnosis(symptoms, patient_context)
        mock_result["_notice"] = "API Key not configured. Using HealthTrack clinical demonstration engine."
        mock_result["_source"] = "MOCK_DEMO"
        return mock_result

    @retry_with_backoff(max_retries=3, base_delay=0.6, max_delay=3.0, jitter=True)
    def _execute_with_retries(
        self, symptoms: str, patient_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Invokes Gemini API wrapped in exponential backoff and jitter."""
        return self._call_gemini(symptoms, patient_context)

    def _call_gemini(
        self, symptoms: str, patient_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute request against Gemini API with JSON output parsing."""
        user_content = f"PATIENT SYMPTOMS:\n{symptoms}\n"
        if patient_context:
            user_content += f"\nPATIENT CONTEXT:\n"
            for k, v in patient_context.items():
                if v:
                    user_content += f"- {k.replace('_', ' ').title()}: {v}\n"

        models_to_try = [self.model_name, "gemini-3.8-flash", "gemini-flash-latest", "gemini-pro-latest"]
        seen = set()
        candidate_models = [m for m in models_to_try if m and not (m in seen or seen.add(m))]

        raw_text = None
        last_err = None

        for candidate in candidate_models:
            try:
                if GENAI_SDK == "genai":
                    response = self.client.models.generate_content(
                        model=candidate,
                        contents=user_content,
                        config=types.GenerateContentConfig(
                            system_instruction=self.SYSTEM_PROMPT,
                            temperature=0.2,
                            response_mime_type="application/json",
                        ),
                    )
                    raw_text = response.text
                else:
                    model = legacy_genai.GenerativeModel(
                        model_name=candidate,
                        system_instruction=self.SYSTEM_PROMPT,
                        generation_config={
                            "temperature": 0.2,
                            "response_mime_type": "application/json",
                        },
                    )
                    response = model.generate_content(user_content)
                    raw_text = response.text
                
                # If we succeeded, update self.model_name for future calls
                self.model_name = candidate
                break
            except Exception as e:
                last_err = e
                err_str = str(e).lower()
                if "404" in err_str or "not found" in err_str or "not supported" in err_str:
                    continue  # Try next candidate model
                raise e

        if raw_text is None:
            raise last_err or RuntimeError("No compatible Gemini model found.")

        parsed_json = self._parse_json_response(raw_text)
        validated_data = self._ensure_minimum_diagnoses(parsed_json, symptoms)
        validated_data["_source"] = f"GEMINI_API ({self.model_name})"
        return validated_data

    def _parse_json_response(self, text: str) -> Dict[str, Any]:
        """Sanitize and parse JSON from model response text."""
        cleaned = text.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if match:
                return json.loads(match.group(0))
            raise ValueError("Failed to parse valid JSON from LLM response.")

    def _ensure_minimum_diagnoses(
        self, data: Dict[str, Any], symptoms: str
    ) -> Dict[str, Any]:
        """Guarantee at least 3 diagnoses exist in the output."""
        diagnoses = data.get("diagnoses", [])
        if not isinstance(diagnoses, list):
            diagnoses = []

        while len(diagnoses) < 3:
            supp_index = len(diagnoses) + 1
            diagnoses.append({
                "name": f"Secondary Diagnostic Consideration #{supp_index}",
                "likelihood": "Low",
                "rationale": "Requires clinical exclusion based on symptomatic overlap with acute presentations.",
                "treatments": [
                    "Perform targeted clinical reassessment",
                    "Conduct basic laboratory evaluation",
                    "Supportive hydration and monitoring",
                ],
                "urgency": "Routine",
                "recommended_tests": ["Complete Blood Count", "Basic Metabolic Panel"],
            })

        data["diagnoses"] = diagnoses
        if "immediate_recommendations" not in data:
            data["immediate_recommendations"] = [
                "Verify vital signs (BP, HR, RR, SpO2, Temp).",
                "Perform physical examination focused on presenting complaint.",
            ]
        if "red_flags" not in data:
            data["red_flags"] = [
                "Sudden severe worsening of symptoms",
                "Hemodynamic instability or altered mental status",
            ]
        if "disclaimer" not in data:
            data["disclaimer"] = (
                "HealthTrack AI is an assistive decision-support tool for licensed healthcare professionals. "
                "Always verify findings with direct clinical examination."
            )

        return data

    def _generate_fallback_diagnosis(
        self, symptoms: str, patient_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Intelligent clinical heuristic engine for offline/demo operation.
        Returns high-quality, realistic differential diagnoses based on symptom keywords.
        """
        s_lower = symptoms.lower()

        # Chest Pain presentation
        if any(w in s_lower for w in ["chest pain", "angina", "arm pain", "shortness of breath", "myocardial", "palpitations"]):
            return {
                "diagnoses": [
                    {
                        "name": "Acute Coronary Syndrome (ACS / Unstable Angina / NSTEMI)",
                        "likelihood": "High",
                        "rationale": "Chest pain with potential radiation, dyspnea, or pressure-like discomfort is classic for myocardial ischemia and demands immediate cardiac triage.",
                        "treatments": [
                            "Immediate 12-lead ECG within 10 minutes",
                            "Chewable Aspirin 162-325 mg (if no contraindication)",
                            "Sublingual Nitroglycerin 0.4 mg q5min (if SBP > 90 mmHg and no RV infarct)",
                            "Serial Cardiac Troponins (at presentation and 1-3 hours)",
                            "Urgent Cardiology / Cath Lab consultation",
                        ],
                        "urgency": "Emergency",
                        "recommended_tests": ["12-Lead ECG", "High-Sensitivity Troponin I/T", "Chest Radiograph (CXR)", "Basic Metabolic Panel"],
                    },
                    {
                        "name": "Gastroesophageal Reflux Disease (GERD) / Esophageal Spasm",
                        "likelihood": "Moderate",
                        "rationale": "Retrosternal discomfort exacerbated by meals or lying supine can closely mimic cardiac angina.",
                        "treatments": [
                            "Trial of Proton Pump Inhibitor (e.g., Omeprazole 20-40 mg daily)",
                            "Antacid / H2 blocker for immediate symptomatic relief",
                            "Dietary and positional modifications (avoid lying down 3 hrs post-prandial)",
                        ],
                        "urgency": "Routine",
                        "recommended_tests": ["Upper Endoscopy (EGD) if refractory", "H. pylori stool antigen"],
                    },
                    {
                        "name": "Costochondritis / Musculoskeletal Chest Wall Pain",
                        "likelihood": "Low",
                        "rationale": "Localized anterior chest wall tenderness reproducible on palpation over costosternal joints.",
                        "treatments": [
                            "Oral NSAIDs (e.g., Ibuprofen 400-600 mg TID with food)",
                            "Local heat or ice application",
                            "Avoidance of strenuous upper-body lifting",
                        ],
                        "urgency": "Routine",
                        "recommended_tests": ["Physical exam with reproducible focal tenderness"],
                    },
                ],
                "immediate_recommendations": [
                    "Obtain immediate 12-lead ECG and apply continuous telemetry monitoring.",
                    "Establish IV access and check vital signs every 15 minutes.",
                    "Administer supplemental oxygen only if SpO2 < 90%.",
                ],
                "red_flags": [
                    "Diaphoresis, syncope, or presyncope",
                    "Radiation of pain to neck, jaw, or both arms",
                    "Hemodynamic instability (Hypotension SBP < 90, ventricular arrhythmias)",
                ],
                "disclaimer": "HealthTrack AI decision support. ACS must be definitively ruled out before entertaining benign etiologies.",
            }

        # Abdominal Pain presentation
        if any(w in s_lower for w in ["abdominal", "stomach pain", "right lower quadrant", "rlq", "nausea", "vomiting", "appendix"]):
            return {
                "diagnoses": [
                    {
                        "name": "Acute Appendicitis",
                        "likelihood": "High",
                        "rationale": "Abdominal pain migrating or localizing to the Right Lower Quadrant (McBurney's point) accompanied by anorexia, nausea, or low-grade fever.",
                        "treatments": [
                            "Keep patient NPO (nil per os) for surgical evaluation",
                            "IV fluid resuscitation (Normal Saline or Lactated Ringer's)",
                            "Pre-operative broad-spectrum IV antibiotics (e.g., Ceftriaxone + Metronidazole)",
                            "Immediate General Surgery consultation for appendectomy",
                        ],
                        "urgency": "Emergency",
                        "recommended_tests": ["Abdominal/Pelvic CT with IV contrast (or Ultrasound in pediatric/pregnant)", "Complete Blood Count (Leukocytosis)", "CRP"],
                    },
                    {
                        "name": "Acute Gastroenteritis (Viral / Bacterial)",
                        "likelihood": "Moderate",
                        "rationale": "Diffuse crampy abdominal discomfort, nausea, vomiting, or diarrhea without focal peritoneal signs.",
                        "treatments": [
                            "Oral rehydration solutions or IV crystalloid rehydration",
                            "Antiemetic therapy (Ondansetron 4-8 mg IV/PO)",
                            "BRAT diet once tolerating oral intake",
                        ],
                        "urgency": "Routine",
                        "recommended_tests": ["Basic Metabolic Panel (Electrolytes/Creatinine)", "Stool PCR/culture if persistent fever"],
                    },
                    {
                        "name": "Mesenteric Adenitis / Ovarian Pathology (if applicable)",
                        "likelihood": "Low",
                        "rationale": "Self-limiting inflammation of mesenteric lymph nodes (often post-viral) or pelvic/adnexal etiology mimicking appendicitis.",
                        "treatments": [
                            "Supportive hydration and analgesia (Acetaminophen)",
                            "Pelvic ultrasound to rule out ovarian torsion/cyst in female patients",
                            "Outpatient observation if CT/ultrasound is negative",
                        ],
                        "urgency": "Urgent",
                        "recommended_tests": ["Pelvic/Transvaginal Ultrasound", "Urine Pregnancy Test (hCG)"],
                    },
                ],
                "immediate_recommendations": [
                    "Maintain NPO status until acute surgical abdomen is formally ruled out.",
                    "Check abdominal examination for involuntary guarding, rebound tenderness, or Rovsing sign.",
                ],
                "red_flags": [
                    "Rigid board-like abdomen (Peritonitis)",
                    "High fever > 38.5°C with severe localized peritoneal signs",
                    "Inability to tolerate oral fluids leading to severe dehydration",
                ],
                "disclaimer": "HealthTrack AI decision support. Acute abdomen requires serial physical examinations and imaging.",
            }

        # Headache / Neurological presentation
        if any(w in s_lower for w in ["headache", "migraine", "photophobia", "neck stiffness", "head ache"]):
            return {
                "diagnoses": [
                    {
                        "name": "Acute Migraine with/without Aura",
                        "likelihood": "High",
                        "rationale": "Unilateral or throbbing headache associated with nausea, photophobia, or phonophobia, typically lasting 4-72 hours.",
                        "treatments": [
                            "Triptan therapy (e.g., Sumatriptan 50-100 mg PO or 6 mg SC at onset)",
                            "NSAID (Naproxen 500 mg or Ketorolac 30 mg IV/IM)",
                            "Antiemetic adjunct (Metoclopramide 10 mg IV)",
                            "Rest in a quiet, dark environment with adequate hydration",
                        ],
                        "urgency": "Urgent",
                        "recommended_tests": ["Neurological examination (cranial nerves, motor, sensory, cerebellar)", "Brain MRI/CT if red flags present"],
                    },
                    {
                        "name": "Tension-Type Headache (TTH)",
                        "likelihood": "Moderate",
                        "rationale": "Bilateral, non-pulsating band-like pressure of mild to moderate intensity without severe nausea or disabling vomiting.",
                        "treatments": [
                            "Simple analgesics (Acetaminophen 1000 mg or Ibuprofen 400 mg)",
                            "Stress reduction techniques, hydration, and neck massage",
                            "Avoid frequent analgesic overuse to prevent medication overuse headache",
                        ],
                        "urgency": "Routine",
                        "recommended_tests": ["Fundoscopic examination", "Cervical spine assessment"],
                    },
                    {
                        "name": "Secondary Headache / Meningitis or Subarachnoid Hemorrhage",
                        "likelihood": "Low",
                        "rationale": "Must be excluded if sudden 'thunderclap' onset, fever, neck stiffness (meningismus), or focal neurologic deficits are present.",
                        "treatments": [
                            "Immediate Non-contrast Head CT (rule out SAH)",
                            "Lumbar Puncture (CSF analysis for opening pressure, cell count, culture)",
                            "Empiric IV antibiotics (Ceftriaxone + Vancomycin) and Dexamethasone if meningitis suspected",
                        ],
                        "urgency": "Emergency",
                        "recommended_tests": ["Emergent Non-contrast Head CT", "Lumbar Puncture with CSF Panel", "Blood Cultures x2"],
                    },
                ],
                "immediate_recommendations": [
                    "Assess for Kernig's and Brudzinski's signs and check pupillary symmetry.",
                    "Inquire specifically about maximal pain intensity at onset ('thunderclap').",
                ],
                "red_flags": [
                    "Sudden instantaneous onset reaching 10/10 severity in seconds",
                    "Nuchal rigidity, high fever, or altered mental status",
                    "New onset neurological focal weakness or speech difficulty",
                ],
                "disclaimer": "HealthTrack AI decision support. Always screen for SNOOP4 secondary headache red flags.",
            }

        # Default / General Respiratory or Systemic Presentation
        return {
            "diagnoses": [
                {
                    "name": "Upper Respiratory Tract Infection (Viral Pharyngitis / Rhinovirus / Flu)",
                    "likelihood": "High",
                    "rationale": "Constellation of symptoms reflects viral mucosal inflammation of the upper aerodigestive tract.",
                    "treatments": [
                        "Supportive hydration and saline gargles",
                        "Analgesics / Antipyretics (Acetaminophen 650 mg q6h or Ibuprofen 400 mg q8h PRN)",
                        "Decongestants or nasal saline sprays for symptom relief",
                        "Patient education: Avoid inappropriate antibiotic usage for viral syndromes",
                    ],
                    "urgency": "Routine",
                    "recommended_tests": ["Rapid Multiplex Viral PCR Panel (COVID-19 / Influenza A+B / RSV)", "Rapid Strep antigen test if sore throat dominant"],
                },
                {
                    "name": "Community-Acquired Bronchitis or Early Pneumonia",
                    "likelihood": "Moderate",
                    "rationale": "Persistent cough with lower tract symptoms, bronchial irritation, or localized adventitious sounds.",
                    "treatments": [
                        "Inhaled beta-2 agonist (Albuterol MDI 2 puffs q4-6h) if bronchospasm present",
                        "Oral Macrolide or Amoxicillin/Clavulanate if bacterial consolidation confirmed",
                        "Adequate humidification and pulmonary toilet",
                    ],
                    "urgency": "Urgent",
                    "recommended_tests": ["Chest Radiograph (PA and Lateral)", "Pulse Oximetry continuous monitoring", "Complete Blood Count"],
                },
                {
                    "name": "Allergic Rhinosinusitis / Environmental Hypersensitivity",
                    "likelihood": "Low",
                    "rationale": "Non-infectious mucosal hypersensitivity triggered by aeroallergens with clear rhinorrhea, sneezing, or itchy eyes.",
                    "treatments": [
                        "Second-generation oral antihistamine (Cetirizine 10 mg or Loratadine 10 mg daily)",
                        "Intranasal corticosteroid spray (Fluticasone propionate 1-2 sprays per nostril daily)",
                        "Environmental trigger avoidance and HEPA air filtration",
                    ],
                    "urgency": "Routine",
                    "recommended_tests": ["Serum specific IgE testing (if chronic)", "Anterior rhinoscopy"],
                },
            ],
            "immediate_recommendations": [
                "Measure core temperature, respiratory rate, and pulse oximetry.",
                "Auscultate bilateral lung fields for crackles, wheezes, or decreased breath sounds.",
            ],
            "red_flags": [
                "Stridor, accessory muscle use, or inability to speak full sentences",
                "Oxygen saturation falling below 92% on ambient air",
                "Altered mental status or progressive lethargy",
            ],
            "disclaimer": "HealthTrack AI decision support. Clinical judgment and in-person patient physical evaluation remain mandatory.",
        }
