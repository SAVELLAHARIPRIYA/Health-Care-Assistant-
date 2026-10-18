"""
HealthTrack AI: Patient Symptom Analysis Tool
Clinical Decision Support System for Healthcare Professionals (Doctors & Nurses)

Powered by Google Gemini AI with Caching, Audit Logging, Input Validation,
and Enterprise Resilience (Circuit Breaker, Exponential Backoff + Jitter).
"""

import json
import os
import sys
import time
from datetime import datetime
from dotenv import load_dotenv
import streamlit as st

# Ensure modules directory is importable
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from modules.cache_service import SymptomCache
from modules.llm_service import ClinicalAnalysisService
from modules.logger_service import AuditLogger
from modules.validator import SymptomValidator

# Load environment variables
load_dotenv()

# Streamlit Page Configuration
st.set_page_config(
    page_title="HealthTrack AI - Clinical Symptom Analysis",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize persistent session singletons
if "cache_service" not in st.session_state:
    st.session_state.cache_service = SymptomCache()

if "logger_service" not in st.session_state:
    st.session_state.logger_service = AuditLogger()

# Custom Modern Clinical CSS Styling
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3rem;
    }

    .clinic-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0369a1 100%);
        color: white;
        padding: 24px 28px;
        border-radius: 14px;
        margin-bottom: 24px;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.15);
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .clinic-header h1 {
        margin: 0;
        font-size: 26px;
        font-weight: 700;
        color: #f8fafc;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .clinic-header p {
        margin: 6px 0 0 0;
        color: #94a3b8;
        font-size: 14px;
    }

    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 12px;
        font-weight: 600;
    }
    .status-pill.online {
        background-color: #dcfce7;
        color: #15803d;
        border: 1px solid #86efac;
    }
    .status-pill.resilience {
        background-color: #f0fdf4;
        color: #166534;
        border: 1px solid #bbf7d0;
    }

    .diagnosis-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 18px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
        border-left: 5px solid #0284c7;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .diagnosis-card:hover {
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08);
    }
    .diagnosis-card.emergency { border-left-color: #ef4444; }
    .diagnosis-card.urgent { border-left-color: #f59e0b; }
    .diagnosis-card.routine { border-left-color: #10b981; }

    .card-title-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 10px;
        flex-wrap: wrap;
        gap: 8px;
    }
    .diagnosis-title {
        font-size: 18px;
        font-weight: 700;
        color: #0f172a;
        margin: 0;
    }
    .badge-group {
        display: flex;
        gap: 6px;
    }
    .badge {
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 0.3px;
        text-transform: uppercase;
    }
    .badge-emergency { background: #fee2e2; color: #b91c1c; }
    .badge-urgent { background: #fef3c7; color: #b45309; }
    .badge-routine { background: #d1fae5; color: #047857; }
    .badge-high { background: #ede9fe; color: #6d28d9; }
    .badge-moderate { background: #e0e7ff; color: #3730a3; }
    .badge-low { background: #f1f5f9; color: #475569; }

    .clinical-rationale {
        background: #f8fafc;
        border-left: 3px solid #cbd5e1;
        padding: 10px 14px;
        border-radius: 4px;
        font-size: 13.5px;
        color: #334155;
        margin: 10px 0 14px 0;
        line-height: 1.5;
    }

    .treatment-item {
        display: flex;
        align-items: baseline;
        gap: 8px;
        font-size: 13.5px;
        color: #1e293b;
        margin-bottom: 6px;
    }
    .treatment-check {
        color: #0284c7;
        font-weight: bold;
    }

    .red-flags-box {
        background: #fff1f2;
        border: 1px solid #fecdd3;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 20px;
    }
    .red-flags-title {
        color: #be123c;
        font-weight: 700;
        font-size: 15px;
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 8px;
    }

    .medical-disclaimer {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 12px 16px;
        color: #64748b;
        font-size: 12px;
        margin-top: 24px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# SIDEBAR CONFIGURATION, RESILIENCE & PRESETS
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🩺 HealthTrack AI")
    st.caption("Clinical Decision Support Tool · v1.3")
    st.divider()



    # Caching Layer Configuration
    st.markdown("#### ⚡ Caching Layer")
    enable_cache = st.toggle("Enable Symptom Cache", value=True, help="Cache repeated queries to ensure < 2s response time.")
    cache_stats = st.session_state.cache_service.get_stats()

    col_c1, col_c2 = st.columns(2)
    col_c1.metric("Cached", f"{cache_stats['total_cached_items']}")
    col_c2.metric("Hits / Misses", f"{cache_stats['hits']}/{cache_stats['misses']}")

    if st.button("🗑️ Clear Cache", use_container_width=True):
        st.session_state.cache_service.clear()
        st.toast("Symptom cache cleared successfully!", icon="🧹")
        st.rerun()

    st.divider()

    # Reliability & Resilience Monitor
    st.markdown("#### 🛡️ Reliability & Resilience")
    temp_llm = ClinicalAnalysisService()
    cb_status = temp_llm.get_resilience_status()
    cb_state = cb_status["state"]

    if cb_state == "CLOSED":
        st.success(f"🟢 **Circuit Breaker: CLOSED** (Healthy)", icon="🛡️")
    elif cb_state == "OPEN":
        st.error(
            f"🔴 **Circuit Breaker: OPEN** (Fail-Fast Active · {cb_status['remaining_cooldown_sec']}s cooldown)",
            icon="🚨",
        )
    else:
        st.warning(f"🟡 **Circuit Breaker: HALF-OPEN** (Trial Probing)", icon="⚠️")

    col_r1, col_r2 = st.columns(2)
    col_r1.caption(f"Failures: {cb_status['consecutive_failures']}/{cb_status['failure_threshold']}")
    col_r2.caption(f"Total Trips: {cb_status['total_trips']}")

    if cb_state != "CLOSED":
        if st.button("🔄 Reset Circuit Breaker", use_container_width=True):
            temp_llm.reset_circuit_breaker()
            st.toast("Circuit Breaker reset to CLOSED!", icon="✅")
            st.rerun()

    st.divider()
    st.caption("HealthTrack AI Diagnostic Assistant · Python / Streamlit / Gemini LLM")

# -----------------------------------------------------------------------------
# MAIN APP HEADER
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="clinic-header">
        <div>
            <h1><span>🩺</span> HealthTrack AI</h1>
            <p>Clinical Decision Support & Patient Symptom Differential Diagnosis</p>
        </div>
        <div style="display: flex; gap: 8px;">
            <span class="status-pill online">● System Operational</span>
            <span class="status-pill resilience">🛡️ Resilience Active</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Application Navigation Tabs
tab_assessment, tab_logs, tab_cache_inspect, tab_tradeoffs = st.tabs([
    "📋 Symptom Analysis",
    "📜 Audit & Compliance Logs",
    "⚡ Cache Monitor",
    "⚖️ Architecture & Resilience",
])

# -----------------------------------------------------------------------------
# TAB 1: CLINICAL SYMPTOM ASSESSMENT (PRIMARY INTERFACE)
# -----------------------------------------------------------------------------
with tab_assessment:
    st.markdown("##### 🩺 Patient Clinical Presentation Form")
    st.caption("Enter patient-reported symptoms, physical observations, and triage findings below.")

    # Optional Patient Demographic Metadata
    with st.expander("👤 Optional Patient Demographics & Baseline Context", expanded=False):
        p_col1, p_col2, p_col3 = st.columns(3)
        with p_col1:
            age_group = st.selectbox(
                "Age Bracket",
                ["Adult (18-64)", "Pediatric (< 12)", "Adolescent (12-17)", "Geriatric (65+)"],
                index=0,
            )
        with p_col2:
            patient_sex = st.selectbox("Biological Sex", ["Unspecified", "Female", "Male"], index=0)
        with p_col3:
            symptom_duration = st.selectbox(
                "Symptom Duration",
                ["< 24 Hours (Hyperacute)", "1-3 Days (Acute)", "1-4 Weeks (Subacute)", "> 1 Month (Chronic)"],
                index=1,
            )

    # Symptom Input Form (Must-have requirement)
    with st.form("symptom_assessment_form", clear_on_submit=False):
        symptom_input = st.text_area(
            "Patient Symptoms Description",
            value="",
            height=140,
            placeholder="E.g., 42-year-old patient complaining of severe persistent headache with photophobia and nausea for 2 days. Denies stiff neck or visual changes.",
            help="Describe presenting symptoms, timeline, severity, and pertinent clinical negatives.",
        )

        char_count = len(symptom_input.strip())
        col_btn, col_info = st.columns([1, 4])
        with col_btn:
            submit_button = st.form_submit_button(
                "🔍 Analyze Symptoms",
                type="primary",
                use_container_width=True,
            )
        with col_info:
            st.caption(f"Character count: {char_count} (Minimum required: 10)")

    # -------------------------------------------------------------------------
    # PROCESSING SUBMISSION
    # -------------------------------------------------------------------------
    if submit_button:
        # Step 1: Input Validation (Should-have requirement)
        validation = SymptomValidator.validate(symptom_input)

        if not validation.is_valid:
            st.error(
                f"⚠️ **Validation Failed:** {validation.error_message}",
                icon="🚫",
            )
            st.session_state.logger_service.log_assessment(
                symptoms=symptom_input or "(empty)",
                output_data={},
                source="VALIDATION_GUARD",
                status="ERROR",
                error_message=validation.error_message,
            )
        else:
            # Step 2: Valid input accepted, display loading indicator (Must-have requirement)
            sanitized_symptoms = validation.sanitized_text
            patient_context = {
                "age_bracket": age_group,
                "biological_sex": patient_sex,
                "duration": symptom_duration,
            }

            start_time = time.time()
            result_data = None
            data_source = "LIVE_LLM"
            is_cached = False

            # Check Caching Layer (Could-have requirement)
            if enable_cache:
                cached_entry = st.session_state.cache_service.get(
                    sanitized_symptoms, patient_context
                )
                if cached_entry:
                    result_data, cached_ts = cached_entry
                    data_source = "CACHE"
                    is_cached = True

            # If not in cache, call Gemini LLM with Circuit Breaker & Retries
            if not result_data:
                with st.spinner("🤖 Analyzing symptoms with Gemini Clinical AI (Resilience Protection Active)..."):
                    llm = ClinicalAnalysisService()
                    result_data = llm.analyze_symptoms(
                        sanitized_symptoms, patient_context
                    )
                    data_source = result_data.get("_source", "GEMINI_API")

                    # Store in cache if enabled
                    if enable_cache:
                        st.session_state.cache_service.set(
                            sanitized_symptoms, result_data, patient_context
                        )

            elapsed_time = time.time() - start_time

            # Step 3: Record input and output in audit log text file (Should-have requirement)
            query_id = st.session_state.logger_service.log_assessment(
                symptoms=sanitized_symptoms,
                output_data=result_data,
                source=data_source,
                response_time_sec=elapsed_time,
                patient_context=patient_context,
                status="SUCCESS",
            )

            # Store in session state for rendering & persistence
            st.session_state.current_result = {
                "data": result_data,
                "query_id": query_id,
                "symptoms": sanitized_symptoms,
                "elapsed": elapsed_time,
                "source": data_source,
                "is_cached": is_cached,
            }

    # -------------------------------------------------------------------------
    # DISPLAY RESULTS IN USER-FRIENDLY FORMAT (Must-have requirement)
    # -------------------------------------------------------------------------
    if "current_result" in st.session_state:
        res = st.session_state.current_result
        data = res["data"]
        diagnoses = data.get("diagnoses", [])
        immediate_recs = data.get("immediate_recommendations", [])
        red_flags = data.get("red_flags", [])
        disclaimer = data.get("disclaimer", "")
        notice = data.get("_notice")

        st.divider()

        # Latency & Source Banner
        b_col1, b_col2 = st.columns([3, 1])
        with b_col1:
            if res["is_cached"]:
                st.success(
                    f"⚡ **Retrieved from Cache** in **{res['elapsed']:.4f}s** (SLA < 2.0s satisfied) · Query ID: `{res['query_id']}`",
                    icon="⚡",
                )
            else:
                source_label = "Gemini LLM (Live API)" if "GEMINI" in res["source"] else "Clinical Fallback Engine (Resilience Active)"
                st.info(
                    f"🤖 **Generated via {source_label}** in **{res['elapsed']:.2f}s** · Query ID: `{res['query_id']}`",
                    icon="🩺",
                )
        with b_col2:
            st.download_button(
                "📥 Export Assessment",
                data=json.dumps(data, indent=2),
                file_name=f"healthtrack_assessment_{res['query_id']}.json",
                mime="application/json",
                use_container_width=True,
            )

        if notice:
            st.info(notice, icon="🛡️")

        # Red Flags / Urgent Triage Banner
        if red_flags:
            st.markdown(
                f"""
                <div class="red-flags-box">
                    <div class="red-flags-title">🚨 Immediate Red Flags & Alarm Symptoms</div>
                    <ul style="margin: 0; padding-left: 20px; color: #9f1239; font-size: 13.5px;">
                        {''.join(f'<li>{flag}</li>' for flag in red_flags)}
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Section Header: Differential Diagnoses (Must contain >= 3)
        st.markdown(f"### 📋 Differential Diagnoses & Evidence-Based Treatments ({len(diagnoses)} Identified)")
        st.caption("Potential conditions ranked by clinical likelihood based on patient presentation.")

        # Render each diagnosis card
        for i, diag in enumerate(diagnoses, 1):
            name = diag.get("name", f"Diagnosis #{i}")
            likelihood = diag.get("likelihood", "Moderate")
            urgency = diag.get("urgency", "Routine")
            rationale = diag.get("rationale", "")
            treatments = diag.get("treatments", [])
            tests = diag.get("recommended_tests", [])

            urgency_class = f"badge-{urgency.lower()}" if urgency.lower() in ["emergency", "urgent", "routine"] else "badge-routine"
            likelihood_class = f"badge-{likelihood.lower()}" if likelihood.lower() in ["high", "moderate", "low"] else "badge-moderate"
            card_border_class = urgency.lower() if urgency.lower() in ["emergency", "urgent", "routine"] else "routine"

            card_html = f"""
            <div class="diagnosis-card {card_border_class}">
                <div class="card-title-row">
                    <h4 class="diagnosis-title">{i}. {name}</h4>
                    <div class="badge-group">
                        <span class="badge {likelihood_class}">Likelihood: {likelihood}</span>
                        <span class="badge {urgency_class}">{urgency}</span>
                    </div>
                </div>
                <div class="clinical-rationale">
                    <strong>Clinical Rationale:</strong> {rationale}
                </div>
            """
            st.markdown(card_html, unsafe_allow_html=True)

            t_col1, t_col2 = st.columns([3, 2])
            with t_col1:
                st.markdown("**💊 Recommended Treatment Suggestions:**")
                if treatments:
                    for t in treatments:
                        st.markdown(
                            f"<div class='treatment-item'><span class='treatment-check'>✓</span> <span>{t}</span></div>",
                            unsafe_allow_html=True,
                        )
                else:
                    st.caption("No specific treatments specified.")

            with t_col2:
                st.markdown("**🔬 Recommended Diagnostic Tests:**")
                if tests:
                    for test in tests:
                        st.markdown(f"- 🧪 `{test}`")
                else:
                    st.caption("Standard vital sign monitoring.")

            st.markdown("</div><br/>", unsafe_allow_html=True)

        if immediate_recs:
            with st.expander("⚡ Immediate Clinical Recommendations & Monitoring", expanded=True):
                for rec in immediate_recs:
                    st.markdown(f"- **{rec}**")

        if disclaimer:
            st.markdown(
                f"""
                <div class="medical-disclaimer">
                    ⚠️ <strong>Clinical Disclaimer:</strong> {disclaimer}
                </div>
                """,
                unsafe_allow_html=True,
            )

# -----------------------------------------------------------------------------
# TAB 2: AUDIT & COMPLIANCE LOGS (Should-have requirement inspection)
# -----------------------------------------------------------------------------
with tab_logs:
    st.markdown("### 📜 Clinical Assessment Audit Trail")
    st.caption("Permanent record of user inputs and model outputs saved to `logs/audit_log.txt`.")

    log_path = st.session_state.logger_service.log_file_path
    st.info(f"Target log file path: `{log_path}`")

    recent_logs = st.session_state.logger_service.read_recent_logs(max_entries=15)

    if not recent_logs:
        st.warning("No audit logs recorded yet. Submit a symptom assessment in Tab 1 to generate entries.")
    else:
        st.markdown(f"**Showing the {len(recent_logs)} most recent audit entries:**")
        for idx, entry in enumerate(reversed(recent_logs), 1):
            with st.expander(f"Audit Record #{idx} (Click to expand)", expanded=(idx == 1)):
                st.code(entry, language="text")

# -----------------------------------------------------------------------------
# TAB 3: CACHE MONITOR & SLA VALIDATION (Could-have requirement)
# -----------------------------------------------------------------------------
with tab_cache_inspect:
    st.markdown("### ⚡ Caching Layer Monitor")
    st.caption("Ensures sub-2-second retrieval times for recurrent clinical presentations.")

    c_stats = st.session_state.cache_service.get_stats()
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    m_col1.metric("Cached Queries", c_stats["total_cached_items"])
    m_col2.metric("Cache Hits", c_stats["hits"])
    m_col3.metric("Cache Misses", c_stats["misses"])
    m_col4.metric("Hit Rate", f"{c_stats['hit_rate_pct']}%")

    st.markdown("#### Stored Query Keys")
    cached_entries = st.session_state.cache_service.list_entries()
    if cached_entries:
        st.dataframe(cached_entries, use_container_width=True)
    else:
        st.info("Cache is currently empty.")

# -----------------------------------------------------------------------------
# TAB 4: ARCHITECTURE, TRADE-OFFS & RESILIENCE
# -----------------------------------------------------------------------------
with tab_tradeoffs:
    st.markdown("### ⚖️ System Architecture, Trade-Offs & Resilience")
    st.caption("Formal documentation of design choices and enterprise reliability patterns.")

    st.markdown(
        """
        #### 1. Architectural Trade-Offs: Caching vs Real-Time Processing
        In clinical decision support systems like **HealthTrack AI**, implementing a caching layer introduces significant benefits alongside critical clinical trade-offs:

        | Dimension | Caching Layer (Implemented) | Pure Real-Time Processing |
        | :--- | :--- | :--- |
        | **Latency & SLA** | **< 10ms retrieval time** (guarantees sub-2s SLA for emergency triaging) | **1.5s – 4.0s** API latency depending on network and Gemini queue |
        | **Token & API Costs** | Zero recurring API cost for duplicate/frequently queried conditions | Linear cost scaling per submission |
        | **Clinical Nuance** | Risk of serving identical outputs if subtle qualifying symptoms match the same normalized key | Captures the latest contextual nuances and varied prompt details |
        | **Model Updates & Freshness** | Cached outputs reflect the model state at time of caching | Automatically benefits from updated medical guidelines and LLM versions |
        | **PHI / HIPAA Privacy** | Stored inputs/outputs require strict cache encryption and data retention policies | Ephemeral processing without persistent symptom storage |

        #### 2. Reliability & Resilience Patterns
        HealthTrack AI incorporates enterprise fault-tolerance at the LLM integration boundary:
        - **Circuit Breaker**: Prevents hammering a degraded or failing Gemini API. Trips to `OPEN` after 3 consecutive failures, immediately serving the high-fidelity clinical fallback engine without latency.
        - **Exponential Backoff with Full Jitter**: Retries transient 429 (Rate Limit) and 503 (Server Overload) errors with jittered delays ($0.6s \to 1.2s \to 2.4s \times \text{Uniform}(0.5, 1.5)$) to prevent thundering herd bottlenecks.
        - **Token Bucket Rate Limiting**: Client-side throttle ensures the application respects the 15 RPM free-tier quota.
        - **Graceful Degradation / Fallback**: Offline clinical heuristic rules ensure the tool remains usable in clinical environments even during complete internet or cloud outages.
        """
    )
