import os
import time
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from modules.central_hub import CentralHub, ensure_schema
from modules.intelligence_engine import IntelligenceEngine
from modules.patient_manager import PatientManager
from modules.simulation_engine import SimulationEngine
from modules.smart_lab import SmartLab


# -----------------------------
# Page config + styles
# -----------------------------

st.set_page_config(layout="wide", page_title="Safety OS", initial_sidebar_state="expanded")

view = st.query_params.get("view", "")

st.markdown(
    """
<style>
  .stAppDeployButton, [data-testid="stDecoration"], [data-testid="stStatusWidget"], #MainMenu {display: none !important;}
  header {visibility: visible !important; background-color: transparent !important;}
  .block-container {padding-top: 1rem; padding-bottom: 2rem;}

  .patient-banner {
    background-color: #f8fafc;
    border: 1px solid #e2e8f0;
    border-left: 4px solid #334155;
    border-radius: 6px;
    padding: 12px 16px;
    margin-bottom: 12px;
  }
  .section-header {
    font-size: 0.85rem;
    font-weight: 700;
    color: #64748b;
    text-transform: uppercase;
    margin: 0 0 8px 0;
    border-bottom: 1px solid #e2e8f0;
    padding-bottom: 4px;
  }

  .pill {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 700;
    margin-left: 8px;
    border: 1px solid #e2e8f0;
    color: #1f2937;
    background: #f8fafc;
  }
  .pill-thinking { color: #1d4ed8; border-color: #bfdbfe; background: #eff6ff; }

  .alert-box {
    padding: 12px;
    border-radius: 6px;
    margin-bottom: 8px;
    border: 1px solid #e2e8f0;
    background-color: #ffffff;
  }
  .alert-high { border-left: 5px solid #ef4444; background-color: #fef2f2; }
  .alert-medium { border-left: 5px solid #f59e0b; background-color: #fffbeb; }
  .alert-low { border-left: 5px solid #10b981; background-color: #ecfdf5; }

  .alert-title { font-weight: 800; font-size: 1rem; color: #111827; }
  .alert-criteria { color: #374151; font-size: 0.92rem; margin-top: 4px; }

  .vital-row { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 8px; }
  .vital-card {
    min-width: 110px;
    padding: 8px 10px;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    background: #ffffff;
  }
  .vital-label { font-size: 0.72rem; font-weight: 700; color: #64748b; text-transform: uppercase; }
  .vital-value { font-size: 1.15rem; font-weight: 800; color: #111827; }

  div.stButton > button { width: 100%; border-radius: 6px; font-weight: 700; font-size: 0.85rem; padding: 6px 10px; }
  .streamlit-expanderHeader { font-size: 0.95rem !important; font-weight: 600 !important; color: #111827 !important; background-color: #f8fafc; border-radius: 6px; }
</style>
""",
    unsafe_allow_html=True,
)


# -----------------------------
# Session initialization
# -----------------------------

if "hub_obj" not in st.session_state:
    # NOTE: "hub" key is reserved for the hub data dict container.
    # Store the CentralHub wrapper separately to avoid key collisions.
    st.session_state["hub_obj"] = CentralHub(st.session_state)
if "emr_synced" not in st.session_state:
    st.session_state.emr_synced = False

if "patient_store" not in st.session_state:
    st.session_state.patient_store = PatientManager()

if "brain" not in st.session_state:
    st.session_state.brain = IntelligenceEngine(st.session_state).start()
else:
    # ensure worker is running
    try:
        st.session_state.brain.start()
    except Exception:
        pass

if "engine" not in st.session_state:
    st.session_state.engine = SimulationEngine(st.session_state.hub_obj, brain=st.session_state.brain)
else:
    # make sure engine has the current brain
    st.session_state.engine.brain = st.session_state.brain

if "smart_lab" not in st.session_state:
    st.session_state.smart_lab = SmartLab(st.session_state.hub_obj)

if "expanded_labs" not in st.session_state:
    st.session_state.expanded_labs = set()


hub: CentralHub = st.session_state["hub_obj"]
brain: IntelligenceEngine = st.session_state.brain
engine: SimulationEngine = st.session_state.engine
p_store: PatientManager = st.session_state.patient_store
smart_lab: SmartLab = st.session_state.smart_lab


# -----------------------------
# Drain background results
# -----------------------------

results = brain.drain_results(limit=30)
for r in results:
    hub.apply_analysis_result(r)

ctx = ensure_schema(hub.data)


# -----------------------------
# Dialog: All labs flowsheet
# -----------------------------

@st.dialog("EMR Flowsheet")
def show_all_labs_dialog(h_labs: Dict[str, List[Dict[str, Any]]]):
    if not h_labs:
        st.info("No labs available.")
        return

    dates_set = set()
    for _lab_name, records in h_labs.items():
        for rec in records:
            if rec.get("dt"):
                dates_set.add(rec["dt"])

    sorted_dates = sorted(list(dates_set))
    table_rows = []

    for lab_name, records in h_labs.items():
        last_rec = records[-1] if records else {}
        unit = last_rec.get("unit", "")
        ref = last_rec.get("ref_range", "")

        row = {"Test": f"{lab_name} ({unit})" if unit else lab_name, "Reference": ref}
        for d in sorted_dates:
            row[d] = "-"
        for rec in records:
            if rec.get("dt"):
                row[rec["dt"]] = rec.get("value", "-")
        table_rows.append(row)

    st.dataframe(pd.DataFrame(table_rows), hide_index=True, use_container_width=True)


# -----------------------------
# Sidebar controls
# -----------------------------

with st.sidebar:
    st.header("System Controls")

    # (1) patient selection
    st.markdown("**1) Patient**")
    patient_ids = p_store.list_patient_ids()
    default_pid = ctx.get("patient", {}).get("patient_id") or (patient_ids[0] if patient_ids else "P1001")
    pid = st.selectbox("Patient ID", options=patient_ids or ["P1001"], index=(patient_ids.index(default_pid) if default_pid in patient_ids else 0))

    if st.button("Sync EMR Data"):
        # ✅ PatientManager가 patients_db.json에서 읽어온 정규화된 스냅샷
        patient_data = p_store.get_patient_snapshot("P1001")
        if not patient_data:
            st.warning("No patient data found in patients_db.json")
            st.rerun()

        ctx = hub.data
        if not isinstance(ctx, dict):
            ctx = {}

        # 최소 스키마 보장
        ctx.setdefault("patient", {})
        ctx.setdefault("visit", {})
        ctx["visit"].setdefault("vitals", {})
        if isinstance(ctx["visit"]["vitals"], dict):
            ctx["visit"]["vitals"].setdefault("latest", {})
        ctx["visit"].setdefault("transcript", [])
        ctx["visit"].setdefault("labs", {})

        # ✅ 환자 기본 정보(EMR snapshot) 반영
        ctx["patient"]["profile"] = patient_data.get("profile", {})
        ctx["patient"]["problems"] = patient_data.get("problems", [])
        ctx["patient"]["allergies"] = patient_data.get("allergies", [])
        ctx["patient"]["medications"] = patient_data.get("medications", [])
        ctx["patient"]["historical_labs"] = patient_data.get("historical_labs", {})

        # ✅ 핵심: vitals는 Sync에서만 들어가게!
        ctx["visit"]["vitals"]["latest"] = patient_data.get("current_vitals", {})

        hub.data = ctx
        st.session_state.emr_synced = True
        st.rerun()

    st.markdown("---")

    # (2) encounter demo
    st.markdown("**2) Encounter / Demo**")
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Start Simulation", type="primary"):
            ok, msg = engine.load_scenario("acs_autopilot.json")
            if not ok:
                st.error(msg)
            else:
                # auto sync scenario patient if present
                scenario_pid = (engine.scenario_data or {}).get("patient_id") or pid
                patient_data = p_store.get_patient_snapshot(scenario_pid)
                hub.apply_patient_snapshot(patient_data or {})

                engine.start()
                hub.append_event({"type": "simulation.start", "ts": time.time(), "source": "ui", "payload": {"scenario": msg}})
                st.session_state.expanded_labs = set()
                st.rerun()

    with col_b:
        if st.button("Reset Encounter"):
            hub.reset()
            st.session_state.expanded_labs = set()
            st.rerun()

    if st.button("Import Labs (demo)"):
        smart_lab.simulate_external_import()
        hub.append_event({"type": "labs.import", "ts": time.time(), "source": "ui"})
        hub.set_ai_status("thinking")
        brain.enqueue_from_hub(hub, reason="labs.import")
        st.rerun()

    st.markdown("---")

    # (3) manual transcript input (for future AI scribe hook)
    st.markdown("**3) Manual Transcript (scribe stub)**")
    role = st.radio("Role", options=["patient", "doctor"], horizontal=True)
    text = st.text_input("Utterance", placeholder="Type what was said...")
    if st.button("Add line"):
        if text.strip():
            ctx2 = ensure_schema(hub.data)
            ctx2["visit"].setdefault("transcript", []).append({"role": role, "text": text.strip(), "ts": time.time()})
            ctx2.setdefault("log", {}).setdefault("events", []).append({"type": "transcript.add", "ts": time.time(), "source": "ui", "payload": {"role": role}})
            hub.data = ctx2

            hub.set_ai_status("thinking")
            brain.enqueue_from_hub(hub, reason="transcript")
            st.rerun()

    st.caption("LLM calls run only if OPENAI_API_KEY is set in environment.")


# -----------------------------
# Patient banner + vitals strip
# -----------------------------

ctx = ensure_schema(hub.data)

demo = ctx.get("patient", {}).get("profile", {})
pmhx_str = ", ".join(ctx.get("patient", {}).get("problems", []) or [])

p_name = demo.get("name", "Unknown Patient")
p_age = demo.get("age", "-")
p_sex = demo.get("sex", demo.get("gender", "-"))
p_pmhx = pmhx_str if pmhx_str else "Not available"
p_alg = ", ".join(ctx.get("patient", {}).get("allergies", []) or []) or "Not available"

if not st.session_state.emr_synced:
    st.info("No patient loaded. Click **Sync EMR Data** to load patient info, vitals, and labs.")

else:
    st.markdown(
        f"""
    <div class="patient-banner">
    <div style="display:flex; justify-content:space-between; align-items:flex-start; gap:16px;">
        <div>
        <span style="font-size: 1.35rem; font-weight: 800;">{p_name}</span>
        <span style="font-size: 1.0rem; font-weight: 700; color:#334155;">({p_age}/{p_sex})</span><br>
        <span style="font-size: 0.85rem; color: #475569;"><b>PMHx:</b> {p_pmhx} &nbsp;|&nbsp; <b>Allergies:</b> {p_alg}</span>
        </div>
        <div>
        <span class="pill">Safety OS</span>
        </div>
    </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

# vitals strip (compact)
v = ((ctx.get("visit") or {}).get("vitals") or {}).get("latest") or {}
if v:
    vitals_html = ""
    for label, key in [("BP", "bp"), ("HR", "hr"), ("RR", "rr"), ("SpO2", "spo2"), ("Temp", "temp")]:
        val = v.get(key, "-")
        vitals_html += f"<div class='vital-card'><div class='vital-label'>{label}</div><div class='vital-value'>{val}</div></div>"
    st.markdown(f"<div class='vital-row'>{vitals_html}</div>", unsafe_allow_html=True)

st.write("DEBUG synced =", hub.data["ui"]["patient_synced"])
st.write("DEBUG vitals =", hub.data["visit"].get("vitals"))
# -----------------------------
# Main layout
# -----------------------------

col_L, col_R = st.columns([1.25, 1], gap="large")

# -------- Left: Note + Critical Labs --------
with col_L:
    # Clinical Note
    head_l, head_r = st.columns([4, 1])
    with head_l:
        st.markdown('<div class="section-header">Clinical Note</div>', unsafe_allow_html=True)
    with head_r:
        if st.button("Copy Note"):
            # Streamlit currently cannot copy to clipboard without custom JS; leave stub.
            st.toast("(Demo) Copy action triggered.")

    note_key = "clinical_note_text"
    if note_key not in st.session_state:
        st.session_state[note_key] = ctx["intelligence"]["doctor"].get("note", {}).get("text", "")

    # keep in sync if AI updated note
    if results:
        st.session_state[note_key] = ctx["intelligence"]["doctor"].get("note", {}).get("text", "")

    note_val = st.text_area(
        "note",
        value=st.session_state[note_key],
        height=280,
        label_visibility="collapsed",
    )

    # persist manual edits back into hub
    if note_val != ctx["intelligence"]["doctor"].get("note", {}).get("text", ""):
        ctx2 = ensure_schema(hub.data)
        ctx2["intelligence"]["doctor"]["note"]["text"] = note_val
        hub.data = ctx2

    st.write("")

    # ✅ Critical Labs 섹션
    st.markdown('<div class="section-header">Critical Labs</div>', unsafe_allow_html=True)

    # 제목 아래에 'Open full labs' 링크만 하나
    st.link_button("Open full labs", "/?view=labs")

    crit = ((ctx.get("visit") or {}).get("labs") or {}).get("critical") or []
    if not crit:
        st.caption("No critical labs detected.")
    else:
        for item in crit:
            st.write(f"- {item}")


# -------- Right: Alerts + Recommendations + Log/Script --------
with col_R:
    # Status indicator
    triage_status = ctx.get("intelligence", {}).get("triage", {}).get("status", "idle")
    doc_status = ctx.get("intelligence", {}).get("doctor", {}).get("status", "idle")
    thinking = triage_status == "thinking" or doc_status == "thinking"
    status_pill = "<span class='pill pill-thinking'>Analyzing...</span>" if thinking else ""

    st.markdown(f'<div class="section-header">Safety Alerts {status_pill}</div>', unsafe_allow_html=True)

    triage_alerts = ctx.get("intelligence", {}).get("triage", {}).get("alerts", []) or []
    doctor_alerts = ctx.get("intelligence", {}).get("doctor", {}).get("alerts", []) or []

    all_alerts = triage_alerts + doctor_alerts

    if all_alerts:
        for alert in all_alerts:
            sev = (alert.get("severity") or "medium").lower()
            sev_class = "alert-medium"
            if sev.startswith("high"):
                sev_class = "alert-high"
            elif sev.startswith("low"):
                sev_class = "alert-low"

            title = alert.get("title", "Alert")
            criteria = alert.get("criteria", "")

            st.markdown(
                f"<div class='alert-box {sev_class}'>"
                f"<div class='alert-title'>{title}</div>"
                f"<div class='alert-criteria'>{criteria}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
    else:
        st.markdown("<div style='color:#94a3b8; font-size:0.9rem;'>Monitoring...</div>", unsafe_allow_html=True)

    # Recommendations
    recs = ctx.get("intelligence", {}).get("doctor", {}).get("recommendations", []) or []
    if recs:
        st.write("")
        st.markdown('<div class="section-header">Recommendations</div>', unsafe_allow_html=True)
        for r in recs:
            st.markdown(f"- {r}")

    # Suggested Orders
    orders = ctx.get("intelligence", {}).get("doctor", {}).get("suggested_orders", []) or []
    if orders:
        st.write("")
        st.markdown('<div class="section-header">Suggested Orders</div>', unsafe_allow_html=True)
        for i, order in enumerate(orders):
            title = order.get("title", f"Order Group {i+1}")
            priority = order.get("priority")
            header = f"{title}" + (f"  ·  {priority}" if priority else "")
            with st.expander(header, expanded=(i == 0)):
                items = order.get("items") or []
                if items:
                    st.write("\n".join([f"• {x}" for x in items]))
                else:
                    st.caption("(No items)")

    st.write("")
    # --- Current conversation (always visible) ---
    # ✅ Show full Script & Log 하나만 남기고,
    #    expander 바깥에는 "현재 대화(마지막 한 줄)"만 표시합니다.

    transcript = ((ctx.get("visit") or {}).get("transcript") or [])


    # 1) 클릭하면 전체
    with st.expander("Show full Script & Log", expanded=False):
        tab_script, tab_log = st.tabs(["Script", "Log"])

        with tab_script:
            if transcript:
                for msg in transcript:
                    role_lbl = "Provider" if msg.get("role") == "doctor" else "Patient"
                    st.markdown(f"**{role_lbl}**: {msg.get('text','')}")
            else:
                st.caption("Waiting for input...")

        with tab_log:
            events = ((ctx.get("log") or {}).get("events") or [])
            analysis_log = ((ctx.get("log") or {}).get("analysis") or [])

            st.markdown("**Recent events**")
            if events:
                for e in events[-15:]:
                    st.code(str(e), language="json")
            else:
                st.caption("No events.")

            st.markdown("**Recent analyses**")
            if analysis_log:
                for a in analysis_log[-10:]:
                    compact = {
                        "ts": a.get("timestamp"),
                        "reason": a.get("reason"),
                        "triage_alerts": len(((a.get("triage") or {}).get("alerts") or [])),
                        "doctor_alerts": len(((a.get("doctor") or {}).get("alerts") or [])),
                    }
                    st.code(str(compact), language="json")
            else:
                st.caption("No analysis yet.")
    # 2) 기본 화면: 마지막 한 줄만 (타이틀 없이)
    if transcript:
        last_msg = transcript[-1]
        role_lbl = "Provider" if last_msg.get("role") == "doctor" else "Patient"
        st.markdown(f"**{role_lbl}**: {last_msg.get('text','')}")
    else:
        st.caption("Waiting for input...")

# -----------------------------
# Simulation ticking
# -----------------------------

if hub.simulation_start_time and engine.scenario_data:
    engine.update()

    timeline = engine.scenario_data.get("timeline", [])
    if len(hub.processed_event_indices) < len(timeline):
        # keep the demo moving
        time.sleep(1.0)
        st.rerun()
