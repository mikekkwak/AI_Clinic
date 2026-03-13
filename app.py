import os
import time
from typing import Any, Dict, List

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from modules.central_hub import CentralHub, ensure_schema
from modules.intelligence_engine import IntelligenceEngine
from modules.patient_manager import PatientManager
from modules.simulation_engine import SimulationEngine
from modules.smart_lab import SmartLab
from modules.note_engine import NoteEngine


def _parse_dt_any(s: Any) -> Optional[datetime]:
    """
    patients_db.json / labs 데이터의 dt/ts가 여러 형태일 수 있어서 최대한 안전하게 파싱.
    허용 예:
    - "2026-02-12 10:15"
    - "2026-02-12"
    - "02/12/2026"
    - "6 Months Ago" 같은 텍스트 -> None 처리(그래프용 날짜로 부적합)
    """
    if not s:
        return None
    if isinstance(s, datetime):
        return s
    if not isinstance(s, str):
        return None

    s = s.strip()
    lowered = s.lower()
    if "ago" in lowered or "month" in lowered or "day" in lowered or "year" in lowered:
        return None

    fmts = [
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
        "%m/%d/%Y",
        "%m/%d/%Y %H:%M",
        "%m/%d/%Y %H:%M:%S",
    ]
    for f in fmts:
        try:
            return datetime.strptime(s, f)
        except Exception:
            pass
    return None


def _fmt_us_date(dt: Optional[datetime]) -> str:
    """미국식 날짜(MM/DD/YYYY). dt가 없으면 빈 문자열."""
    if not dt:
        return ""
    return dt.strftime("%m/%d/%Y")


def _safe_str(x: Any) -> str:
    if x is None:
        return ""
    return str(x)


def _title_case_order_type(x: str) -> str:
    s = (x or "").strip().replace("_", " ")
    return s.title() if s else ""


def _flatten_suggested_orders_for_ui(ctx: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    doctor output이 아래 두 구조 중 어느 것이 와도 UI에서 일관되게 보여주기 위한 정규화 함수.

    Supported:
    1) old grouped format
       suggested_orders = [
         {"title": "...", "priority": "high", "items": ["ECG", "Troponin"]}
       ]

    2) structured future-ready format
       recommended_orders = [
         {"id": "...", "label": "ECG 12-lead", "type": "procedure", "priority": "high", "reason": "..."}
       ]

    반환 형식:
    [
      {
        "label": "ECG 12-lead",
        "priority": "high",
        "type": "procedure",
        "reason": "Possible ACS",
        "source_group": "Priority Orders"
      }
    ]
    """
    doctor_obj = ((ctx.get("intelligence") or {}).get("doctor") or {})

    normalized: List[Dict[str, Any]] = []

    # Preferred future-ready structured format
    recommended_orders = doctor_obj.get("recommended_orders") or []
    if isinstance(recommended_orders, list) and recommended_orders:
        for row in recommended_orders:
            if not isinstance(row, dict):
                continue
            label = str(row.get("label") or row.get("title") or row.get("id") or "").strip()
            if not label:
                continue
            normalized.append(
                {
                    "label": label,
                    "priority": str(row.get("priority") or "").strip().lower(),
                    "type": str(row.get("type") or "").strip().lower(),
                    "reason": str(row.get("reason") or "").strip(),
                    "source_group": "",
                    "id": str(row.get("id") or "").strip(),
                }
            )

    # Backward compatibility with existing grouped suggested_orders
    if not normalized:
        suggested_orders = doctor_obj.get("suggested_orders") or []
        if isinstance(suggested_orders, list):
            for group in suggested_orders:
                if not isinstance(group, dict):
                    continue
                group_title = str(group.get("title") or "").strip()
                priority = str(group.get("priority") or "").strip().lower()
                items = group.get("items") or []
                if not isinstance(items, list):
                    continue
                for item in items:
                    label = str(item).strip()
                    if not label:
                        continue
                    normalized.append(
                        {
                            "label": label,
                            "priority": priority,
                            "type": "",
                            "reason": "",
                            "source_group": group_title,
                            "id": "",
                        }
                    )

    # Deduplicate while preserving order
    deduped: List[Dict[str, Any]] = []
    seen = set()
    for row in normalized:
        key = (
            row.get("label", "").lower(),
            row.get("priority", "").lower(),
            row.get("type", "").lower(),
            row.get("reason", "").lower(),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)

    # Sort by priority then label
    rank = {"high": 0, "medium": 1, "low": 2, "": 3}
    deduped.sort(key=lambda x: (rank.get(str(x.get("priority") or "").lower(), 9), str(x.get("label") or "").lower()))

    return deduped


def build_labs_table_rows(
    current_items: List[Dict[str, Any]],
    labs_history: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    EMR 스타일 표에 넣을 row 생성.
    Columns:
    - Test
    - Current Result and Flag
    - Previous Result and Date
    - Units
    - (optional) Reference Interval (지금은 빈칸)
    """
    by_name: Dict[str, List[Dict[str, Any]]] = {}
    for h in labs_history or []:
        nm = (h.get("name") or h.get("display") or "").strip()
        if not nm:
            continue
        by_name.setdefault(nm, []).append(h)

    rows: List[Dict[str, Any]] = []

    for item in current_items or []:
        name = (item.get("name") or item.get("display") or "Lab").strip()
        val = item.get("value")
        unit = item.get("unit") or ""
        flag = item.get("flag") or ""

        current_str = f"{_safe_str(val)}"
        if flag:
            current_str += f"   {flag}"

        prev_str = ""
        hist = by_name.get(name, [])

        hist_pts: List[Tuple[Optional[datetime], Dict[str, Any]]] = []
        for h in hist:
            dt = _parse_dt_any(h.get("dt") or h.get("ts") or h.get("date"))
            hist_pts.append((dt, h))
        hist_pts.sort(key=lambda x: (x[0] is None, x[0]))

        cur_dt = _parse_dt_any(item.get("dt") or item.get("ts") or item.get("date"))

        prev_candidate: Optional[Dict[str, Any]] = None

        if hist_pts:
            if cur_dt:
                best_dt = None
                for dt, h in hist_pts:
                    if dt and dt < cur_dt:
                        if best_dt is None or dt > best_dt:
                            best_dt = dt
                            prev_candidate = h
            else:
                if len(hist_pts) >= 2:
                    prev_candidate = hist_pts[-2][1]

        if prev_candidate:
            pv = prev_candidate.get("value")
            pdt = _parse_dt_any(prev_candidate.get("dt") or prev_candidate.get("ts") or prev_candidate.get("date"))
            pdt_s = _fmt_us_date(pdt)
            prev_str = f"{_safe_str(pv)}"
            if pdt_s:
                prev_str += f"  ({pdt_s})"

        rows.append(
            {
                "Test": name,
                "Current Result and Flag": current_str,
                "Previous Result and Date": prev_str,
                "Units": unit,
                "Reference Interval": "",
            }
        )

    return rows


def render_trend_plot(
    test_name: str,
    labs_history: List[Dict[str, Any]],
    current_item: Optional[Dict[str, Any]] = None,
) -> None:
    """
    검사 trend 그래프 표시 (plotly).
    - labs_history의 과거 값 + current_item(오늘 critical 값)을 함께 포함한다.
    """
    pts: List[Tuple[datetime, float]] = []

    for h in labs_history or []:
        nm = (h.get("name") or h.get("display") or "").strip()
        if nm != test_name:
            continue
        dt = _parse_dt_any(h.get("dt") or h.get("ts") or h.get("date"))
        if not dt:
            continue
        val = h.get("value")
        try:
            fv = float(val)
        except Exception:
            continue
        pts.append((dt, fv))

    if current_item:
        dtc = _parse_dt_any(current_item.get("dt") or current_item.get("ts") or current_item.get("date"))
        if dtc:
            try:
                fvc = float(current_item.get("value"))
                pts.append((dtc, fvc))
            except Exception:
                pass

    pts.sort(key=lambda x: x[0])
    if len(pts) < 2:
        st.caption("Not enough historical points to show a trend yet.")
        return

    xs = [_fmt_us_date(p[0]) for p in pts]
    ys = [p[1] for p in pts]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers"))
    fig.update_layout(
        height=240,
        margin=dict(l=10, r=10, t=20, b=10),
        xaxis_title="Date (MM/DD/YYYY)",
        yaxis_title="Value",
    )
    st.plotly_chart(fig, width="stretch")


def normalize_labs_from_snapshot(patient_data: dict):
    """
    patients_db.json에서 가져온 lab 데이터 형태가 조금 달라도
    visit.labs.history / latest / critical 형태로 정리해주는 최소 함수.

    반환: (history_list, latest_list, critical_list)
    """
    history = []
    latest = []
    critical = []

    if not isinstance(patient_data, dict):
        return history, latest, critical

    hist = patient_data.get("historical_labs")
    curr = patient_data.get("current_labs")

    if isinstance(hist, list):
        for row in hist:
            if isinstance(row, dict):
                history.append(row)
    elif isinstance(hist, dict):
        for test_name, rows in hist.items():
            if isinstance(rows, list):
                for r in rows:
                    if isinstance(r, dict):
                        rr = dict(r)
                        rr.setdefault("name", test_name)
                        history.append(rr)

    if isinstance(curr, list):
        for row in curr:
            if isinstance(row, dict):
                history.append(row)
                latest.append(row)

    if not latest and history:
        seen = {}
        for r in history:
            nm = (r.get("name") or r.get("display") or r.get("key") or "").strip()
            if nm:
                seen[nm.lower()] = r
        latest = list(seen.values())

    def norm_row(r):
        name = r.get("name") or r.get("display") or r.get("key") or "Lab"
        value = r.get("value", r.get("val"))
        unit = r.get("unit") or ""
        dt = r.get("dt", r.get("ts") or r.get("date")) or ""
        flag = r.get("flag") or ""
        return {"name": str(name), "value": value, "unit": unit, "flag": flag, "dt": str(dt)}

    history_n = [norm_row(r) for r in history if isinstance(r, dict)]
    latest_n = [norm_row(r) for r in latest if isinstance(r, dict)]

    for r in latest_n:
        f = str(r.get("flag", "")).lower()
        if f in ("high", "critical"):
            critical.append(r)

    return history_n, latest_n, critical


def select_key_labs_for_demo(pid: str, latest_labs: List[Dict[str, Any]]):
    """
    Demo용: 케이스별로 의미 있는 labs만 선택.
    원본 labs는 유지하고, UI 표시용만 필터링한다.
    """
    key_map = {
        "P1001": ["Troponin", "Creatinine"],
        "P1002": ["WBC", "Lactate", "Creatinine"],
        "P1003": ["Urinalysis - Nitrite", "Urinalysis - Leukocyte Esterase", "Urinalysis - WBC"],
    }

    wanted = key_map.get(pid, [])

    selected = []
    for lab in latest_labs:
        name = lab.get("name", "")
        for w in wanted:
            if w.lower() in name.lower():
                selected.append(lab)
                break

    return selected


st.set_page_config(layout="wide", page_title="Safety OS", initial_sidebar_state="expanded")

try:
    view = st.query_params.get("view", "")
except Exception:
    qp = st.experimental_get_query_params()
    view = (qp.get("view") or [""])[0]

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

  .order-row {
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    background: #ffffff;
    padding: 10px 12px;
    margin-bottom: 8px;
  }
  .order-row-top {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
  }
  .order-label {
    font-weight: 700;
    color: #111827;
    font-size: 0.96rem;
  }
  .order-meta {
    font-size: 0.80rem;
    color: #475569;
    margin-top: 4px;
  }
  .order-pill {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 700;
    border: 1px solid #e2e8f0;
    background: #f8fafc;
    color: #1f2937;
    white-space: nowrap;
  }
  .order-priority-high {
    background: #fef2f2;
    border-color: #fecaca;
    color: #b91c1c;
  }
  .order-priority-medium {
    background: #fffbeb;
    border-color: #fde68a;
    color: #b45309;
  }
  .order-priority-low {
    background: #ecfdf5;
    border-color: #a7f3d0;
    color: #047857;
  }
</style>
""",
    unsafe_allow_html=True,
)


# -----------------------------
# Session initialization
# -----------------------------

if "hub_obj" not in st.session_state:
    st.session_state["hub_obj"] = CentralHub(st.session_state)
if "emr_synced" not in st.session_state:
    st.session_state.emr_synced = False

if "patient_store" not in st.session_state:
    st.session_state.patient_store = PatientManager()

if "brain" not in st.session_state:
    st.session_state.brain = IntelligenceEngine(st.session_state).start()
else:
    try:
        st.session_state.brain.start()
    except Exception:
        pass

if "engine" not in st.session_state:
    st.session_state.engine = SimulationEngine(st.session_state.hub_obj, brain=st.session_state.brain)
else:
    st.session_state.engine.brain = st.session_state.brain

if "smart_lab" not in st.session_state:
    st.session_state.smart_lab = SmartLab(st.session_state.hub_obj)

if "expanded_labs" not in st.session_state:
    st.session_state.expanded_labs = set()

if "note_engine" not in st.session_state:
    st.session_state.note_engine = NoteEngine()


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

    st.dataframe(pd.DataFrame(table_rows), hide_index=True, width="stretch")


# -----------------------------
# Sidebar controls
# -----------------------------

with st.sidebar:
    st.header("System Controls")

    st.markdown("**1) Patient**")

    patient_ids = p_store.list_patient_ids()
    if not patient_ids:
        patient_ids = ["P1001"]

    if "selected_pid" not in st.session_state:
        current_pid = (ctx.get("patient") or {}).get("patient_id")
        st.session_state.selected_pid = current_pid if current_pid in patient_ids else patient_ids[0]

    pid = st.selectbox(
        "Patient ID",
        options=patient_ids,
        index=patient_ids.index(st.session_state.selected_pid) if st.session_state.selected_pid in patient_ids else 0,
        key="patient_id_selectbox",
    )

    st.session_state.selected_pid = pid

    if st.button("Sync EMR Data"):
        patient_data = p_store.get_patient_snapshot(pid)
        if not patient_data:
            st.warning(f"No patient data found for pid={pid} in patients_db.json")
            st.rerun()

        ctx = ensure_schema(hub.data)

        ctx.setdefault("patient", {})
        ctx["patient"]["patient_id"] = pid

        profile = patient_data.get("profile") or {}
        gender_val = profile.get("gender") or profile.get("sex") or ""

        ctx["patient"]["profile"] = {
            "name": profile.get("name", "Unknown Patient"),
            "age": profile.get("age"),
            "gender": gender_val,
            "sex": gender_val,
        }

        ctx["patient"]["problems"] = patient_data.get("pmhx") or patient_data.get("problems") or []
        ctx["patient"]["allergies"] = patient_data.get("allergies") or []
        ctx["patient"]["medications"] = patient_data.get("medications") or []
        ctx["patient"]["historical_labs"] = patient_data.get("historical_labs") or {}
        ctx["patient"]["current_labs"] = patient_data.get("current_labs") or []

        ctx.setdefault("visit", {})
        ctx["visit"].setdefault("transcript", [])

        ctx["visit"]["vitals"] = {"latest": {}}
        ctx["visit"]["labs"] = {
            "history": [],
            "latest": [],
            "critical": [],
        }

        emr_vitals = patient_data.get("current_vitals") or {}
        norm_vitals = {
            "bp": emr_vitals.get("bp", ""),
            "hr": emr_vitals.get("hr", ""),
            "rr": emr_vitals.get("rr", ""),
            "spo2": emr_vitals.get("spo2") or emr_vitals.get("spo2(%)") or "",
            "temp": emr_vitals.get("temp") or emr_vitals.get("temp(F)") or "",
        }
        ctx["visit"]["vitals"]["latest"] = {k: v for k, v in norm_vitals.items() if v not in (None, "")}

        hidden = (ctx.get("meta") or {}).get("hidden_start_vitals") or {}
        if hidden:
            ctx["visit"]["vitals"]["latest"] = hidden

        history_n, latest_n, critical_n = normalize_labs_from_snapshot(patient_data)

        ctx["visit"]["labs"]["history"] = history_n
        ctx["visit"]["labs"]["latest"] = latest_n
        ctx["visit"]["labs"]["critical"] = critical_n

        display_labs = select_key_labs_for_demo(pid, latest_n)
        if display_labs:
            ctx["visit"]["labs"]["critical"] = display_labs

        hub.data = ctx
        hub.set_patient_synced(True)
        st.session_state.emr_synced = True
        st.session_state.expanded_labs = set()

        try:
            st.query_params.pop("trend", None)
        except Exception:
            try:
                st.experimental_set_query_params()
            except Exception:
                pass

        hub.set_ai_status("thinking")
        brain.enqueue_from_hub(hub, reason="sync")

        st.rerun()

    st.markdown("---")

    st.markdown("**2) Encounter / Demo**")
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Start Simulation", type="primary"):
            st.session_state.emr_synced = False
            hub.set_patient_synced(False)

            scenario_map = {
                "P1001": "acs_autopilot.json",
                "P1002": "sepsis_appendicitis_demo.json",
                "P1003": "uti_demo.json",
            }

            selected_pid = st.session_state.selected_pid
            scenario_file = scenario_map.get(selected_pid, "acs_autopilot.json")

            ok, msg = engine.load_scenario(scenario_file)
            if not ok:
                st.error(msg)
            else:
                engine.start()
                hub.append_event(
                    {
                        "type": "simulation.start",
                        "ts": time.time(),
                        "source": "ui",
                        "payload": {
                            "scenario": scenario_file,
                            "patient_id": selected_pid
                        }
                    }
                )
                st.session_state.expanded_labs = set()

                try:
                    st.query_params.pop("trend", None)
                except Exception:
                    try:
                        st.experimental_set_query_params()
                    except Exception:
                        pass

                st.rerun()

    with col_b:
        if st.button("Reset Encounter"):
            hub.reset()
            st.session_state.emr_synced = False
            hub.set_patient_synced(False)
            st.session_state.expanded_labs = set()

            if "clinical_note_text" in st.session_state:
                del st.session_state["clinical_note_text"]

            try:
                st.query_params.pop("trend", None)
            except Exception:
                try:
                    st.experimental_set_query_params()
                except Exception:
                    pass

            st.rerun()

    st.markdown("---")

    if False:
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
# Patient banner + main layout
# -----------------------------
ctx = ensure_schema(hub.data)

patient_obj = ctx.get("patient") or {}
demo = patient_obj.get("profile") or {}

pmhx_list = patient_obj.get("problems") or patient_obj.get("pmhx") or []
pmhx_str = ", ".join(pmhx_list) if pmhx_list else ""

p_name = demo.get("name", "Unknown Patient")
p_age = demo.get("age", "-")
p_sex = demo.get("sex") or demo.get("gender") or "-"
p_pmhx = pmhx_str if pmhx_str else "Not available"
p_alg = ", ".join(patient_obj.get("allergies", []) or []) or "Not available"

v = ((ctx.get("visit") or {}).get("vitals") or {}).get("latest") or {}

if "spo2" not in v and "spo2(%)" in v:
    v["spo2"] = v.get("spo2(%)")

if "temp" not in v and "temp(F)" in v:
    v["temp"] = v.get("temp(F)")

if not st.session_state.emr_synced:
    st.info("No patient loaded. Click **Sync EMR Data** to load patient info, vitals, and labs.")
else:
    banner_left, banner_right = st.columns([3.2, 2.0], gap="medium")

    with banner_left:
        st.markdown(
            f"""
            <div class="patient-banner" style="margin-bottom:0;">
              <div>
                <span style="font-size: 1.35rem; font-weight: 800;">{p_name}</span>
                <span style="font-size: 1.0rem; font-weight: 700; color:#334155;">({p_age}/{p_sex})</span>
              </div>
              <div style="font-size: 0.85rem; color: #475569; margin-top:4px;">
                <b>PMHx:</b> {p_pmhx} &nbsp;|&nbsp; <b>Allergies:</b> {p_alg}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with banner_right:
        if v:
            vc1, vc2, vc3, vc4, vc5 = st.columns(5, gap="small")
            vital_cols = [vc1, vc2, vc3, vc4, vc5]

            for col, (label, key) in zip(
                vital_cols,
                [("BP", "bp"), ("HR", "hr"), ("RR", "rr"), ("SpO2", "spo2"), ("Temp", "temp")]
            ):
                val = v.get(key, "-")
                col.markdown(
                    f"""
                    <div class="vital-card">
                    <div class="vital-label">{label}</div>
                    <div class="vital-value">{val}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

# -----------------------------
# Main layout
# -----------------------------
st.write("")
st.write("")
st.write("")
st.write("")
st.write("")

col_L, col_R = st.columns([1.25, 1], gap="large")


# -------- Left: Note + Critical Labs --------
with col_L:
    head_l, head_r = st.columns([3, 1])
    with head_l:
        st.markdown('<div class="section-header">Clinical Note</div>', unsafe_allow_html=True)
    with head_r:
        if st.button("Copy Note"):
            st.toast("(Demo) Copy action triggered.")

    note_template_label = st.selectbox(
        "Preferred Note Template",
        options=["Dr. Kim UC Style", "Standard ED Style"],
        index=0,
        key="preferred_note_template",
    )

    template_map = {
        "Dr. Kim UC Style": "uc_dr_kim",
        "Standard ED Style": "ed_standard",
    }
    template_key = template_map[note_template_label]

    note_key = "clinical_note_text"
    note_template_key = "clinical_note_template_key"

    generated_note = st.session_state.note_engine.compose_note(
        ctx,
        template=template_key,
    )

    if note_key not in st.session_state:
        st.session_state[note_key] = generated_note

    if note_template_key not in st.session_state:
        st.session_state[note_template_key] = template_key

    if st.session_state[note_template_key] != template_key:
        st.session_state[note_template_key] = template_key
        st.session_state[note_key] = generated_note
        ctx2 = ensure_schema(hub.data)
        ctx2["intelligence"]["doctor"]["note"]["text"] = generated_note
        hub.data = ctx2

    if results:
        st.session_state[note_key] = generated_note
        ctx2 = ensure_schema(hub.data)
        ctx2["intelligence"]["doctor"]["note"]["text"] = generated_note
        hub.data = ctx2

    note_val = st.text_area(
        "note",
        value=st.session_state[note_key],
        height=280,
        label_visibility="collapsed",
    )

    if note_val != st.session_state[note_key]:
        st.session_state[note_key] = note_val
        ctx2 = ensure_schema(hub.data)
        ctx2["intelligence"]["doctor"]["note"]["text"] = note_val
        hub.data = ctx2

    st.write("")
    st.write("")
    st.write("")
    st.write("")

    labs_head_l, labs_head_r = st.columns([3, 1])
    with labs_head_l:
        st.markdown('<div class="section-header">Critical Labs</div>', unsafe_allow_html=True)
    with labs_head_r:
        st.link_button("Open full labs", "/?view=labs")

    labs_history = ((ctx.get("visit") or {}).get("labs") or {}).get("history") or []
    crit = ((ctx.get("visit") or {}).get("labs") or {}).get("critical") or []

    if not st.session_state.emr_synced:
        st.caption("Sync EMR to view critical labs.")
    elif not crit:
        st.caption("No critical labs detected.")
    else:
        rows = build_labs_table_rows(crit, labs_history)
        df = pd.DataFrame(rows)

        crit_by_name = {(i.get("name") or i.get("display") or "").strip(): i for i in crit}

        sel = st.dataframe(
            df,
            hide_index=True,
            width="stretch",
            on_select="rerun",
            selection_mode="single-row",
        )

        selected_trend = None
        try:
            if sel and hasattr(sel, "selection") and sel.selection and sel.selection.rows:
                ridx = sel.selection.rows[0]
                selected_trend = str(df.iloc[ridx]["Test"])
        except Exception:
            selected_trend = None

        if selected_trend:
            st.markdown("---")
            st.markdown(f"**{selected_trend} Trend**")
            render_trend_plot(
                selected_trend,
                labs_history,
                current_item=crit_by_name.get(selected_trend),
            )
        else:
            st.caption("Click a row to view trend.")


# -------- Right: Alerts + Recommendations + Orders + Script/Log --------
with col_R:
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

    recs = ctx.get("intelligence", {}).get("doctor", {}).get("recommendations", []) or []
    if recs:
        st.write("")
        st.markdown('<div class="section-header">Recommendations</div>', unsafe_allow_html=True)
        for r in recs:
            st.markdown(f"- {r}")

    flat_orders = _flatten_suggested_orders_for_ui(ctx)
    if flat_orders:
        st.write("")
        orders_head_l, orders_head_r = st.columns([3, 1])
        with orders_head_l:
            st.markdown('<div class="section-header">Suggested Orders</div>', unsafe_allow_html=True)
        with orders_head_r:
            if st.button("Send Orders", key="send_orders_button"):
                st.toast("(Demo) Order routing triggered. Future EMR integration point.")

        for order in flat_orders:
            label = order.get("label") or "Order"
            priority = (order.get("priority") or "").strip().lower()
            reason = order.get("reason") or ""
            order_type = _title_case_order_type(order.get("type") or "")
            source_group = order.get("source_group") or ""

            meta_parts: List[str] = []
            if order_type:
                meta_parts.append(order_type)
            if reason:
                meta_parts.append(reason)
            elif source_group:
                meta_parts.append(source_group)

            pill_class = ""
            if priority == "high":
                pill_class = "order-priority-high"
            elif priority == "medium":
                pill_class = "order-priority-medium"
            elif priority == "low":
                pill_class = "order-priority-low"

            priority_html = ""
            if priority:
                priority_html = f"<span class='order-pill {pill_class}'>{priority.title()}</span>"

            meta_html = ""
            if meta_parts:
                meta_html = f"<div class='order-meta'>{' · '.join(meta_parts)}</div>"

            st.markdown(
                f"""
                <div class="order-row">
                  <div class="order-row-top">
                    <div class="order-label">{label}</div>
                    {priority_html}
                  </div>
                  {meta_html}
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.write("")
    st.write("")
    st.write("")
    st.write("")

    transcript = ((ctx.get("visit") or {}).get("transcript") or [])

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
        time.sleep(1.0)
        st.rerun()