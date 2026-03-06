# modules/central_hub.py
from __future__ import annotations

import copy
import time
import queue
import threading
from typing import Any, Dict, Optional, MutableMapping


# ============================================================
# 1) Session helpers (Streamlit session_state compatible)
# ============================================================

def _unwrap_session(session_state: Any) -> MutableMapping[str, Any]:
    """
    Streamlit의 st.session_state는 dict가 아니라 dict처럼 동작하는 객체입니다.
    여기서는 'dict처럼 쓸 수 있는 MutableMapping'으로 그대로 취급합니다.
    """
    if session_state is None:
        # 방어: None이 들어오면 빈 dict로라도 동작하게
        return {}
    return session_state  # Streamlit SessionStateProxy는 MutableMapping처럼 동작


def get_hub_data(session_state: Any) -> Dict[str, Any]:
    ss = _unwrap_session(session_state)
    data = ss.get("hub_data")
    return ensure_schema(data)


def set_hub_data(session_state: Any, new_data: Dict[str, Any]) -> None:
    ss = _unwrap_session(session_state)
    ss["hub_data"] = ensure_schema(new_data)


# ============================================================
# 2) Background queues (per-session)
# ============================================================

def init_hub_queues(session_state: Any) -> Dict[str, Any]:
    """
    IntelligenceEngine가 사용하는 per-session queue/stop_event를 준비합니다.
    언제 호출돼도 항상 dict를 반환하도록 '절대 None 반환 금지'로 설계합니다.
    """
    ss = _unwrap_session(session_state)

    if "hub_task_queue" not in ss:
        ss["hub_task_queue"] = queue.Queue()
    if "hub_result_queue" not in ss:
        ss["hub_result_queue"] = queue.Queue()
    if "hub_stop_event" not in ss:
        ss["hub_stop_event"] = threading.Event()

    # 반환 형식은 IntelligenceEngine이 기대하는 키 이름으로 고정
    return {
        "task_queue": ss["hub_task_queue"],
        "result_queue": ss["hub_result_queue"],
        "stop_event": ss["hub_stop_event"],
    }


# ============================================================
# 3) Hub schema (the MOST IMPORTANT part)
# ============================================================

def default_hub_data() -> Dict[str, Any]:
    """
    hub.data의 기본 뼈대입니다.
    ⚠️ 여기에서 'ui'를 반드시 넣어줘야 KeyError: 'ui'가 사라집니다.
    """
    return {
        "patient": {
            "patient_id": None,
            "profile": {"name": "Unknown Patient", "age": None, "sex": None},
            "problems": [],
            "allergies": [],
            "medications": [],
            "history_summary": "",
            "historical_labs": {},  # { lab_key: [ {ts, value, unit, ...}, ... ] } 형태 권장
        },
        "visit": {
            "vitals": {"latest": {}},  # vitals는 최신값만 latest에 (sync 전엔 비어있게)
            "transcript": [],           # [{"role": "...", "text":"..."}]
            "labs": {
                "latest": [],           # sync 후 현재값 (표시용)
                "critical": [],         # critical shortlist (표시용)
            },
        },
        "intelligence": {
            "triage": {
                "alerts": [],           # [{"severity","title","criteria",...}]
                "status": "idle",
                "meta": {},
            },
            "doctor": {
                "note": {"text": ""},   # note는 항상 여기!
                "suggested_orders": [],
                "recommendations": [],
                "alerts": [],
                "status": "idle",
                "meta": {},
            },
        },
        "log": {
            "events": [],
            "analysis": [],
        },
        # ✅ UI가 기대하는 키들 (없으면 KeyError 폭탄)
        "ui": {
            "patient_synced": False,
            "audit_log": [],
            "last_transcript_line": "",
            "ai_status": "idle",  # "idle" | "thinking" | "done" 등
        },
        "meta": {
            "created_ts": time.time(),
        },
    }


def ensure_schema(data: Any) -> Dict[str, Any]:
    """
    Best-effort: 어떤 중간상태/구버전 dict가 들어와도
    우리가 기대하는 스키마로 '복구/보강'해줍니다.
    """
    if not isinstance(data, dict):
        data = {}

    base = default_hub_data()

    # 1) top-level 키 보강
    for k, v in base.items():
        if k not in data:
            data[k] = copy.deepcopy(v)

    # 2) patient
    data["patient"].setdefault("patient_id", None)
    data["patient"].setdefault("profile", {})
    data["patient"].setdefault("problems", [])
    data["patient"].setdefault("allergies", [])
    data["patient"].setdefault("medications", [])
    data["patient"].setdefault("history_summary", "")
    data["patient"].setdefault("historical_labs", {})

    # normalize sex key
    prof = data["patient"].setdefault("profile", {})
    if "gender" in prof and "sex" not in prof:
        prof["sex"] = prof.get("gender")
    prof.setdefault("name", "Unknown Patient")
    prof.setdefault("age", None)
    prof.setdefault("sex", None)

    # 3) visit
    data["visit"].setdefault("vitals", {"latest": {}})
    if isinstance(data["visit"].get("vitals"), dict) and "latest" not in data["visit"]["vitals"]:
        # older style: visit.vitals = {...}
        data["visit"]["vitals"] = {"latest": data["visit"]["vitals"]}

    data["visit"].setdefault("transcript", [])
    data["visit"].setdefault("labs", {"latest": [], "critical": []})
    data["visit"]["labs"].setdefault("latest", [])
    data["visit"]["labs"].setdefault("critical", [])

    # 4) intelligence
    data["intelligence"].setdefault("triage", {"alerts": [], "status": "idle", "meta": {}})
    data["intelligence"]["triage"].setdefault("alerts", [])
    data["intelligence"]["triage"].setdefault("status", "idle")
    data["intelligence"]["triage"].setdefault("meta", {})

    data["intelligence"].setdefault("doctor", {})
    data["intelligence"]["doctor"].setdefault("note", {"text": ""})
    data["intelligence"]["doctor"]["note"].setdefault("text", "")
    data["intelligence"]["doctor"].setdefault("suggested_orders", [])
    data["intelligence"]["doctor"].setdefault("recommendations", [])
    data["intelligence"]["doctor"].setdefault("alerts", [])
    data["intelligence"]["doctor"].setdefault("status", "idle")
    data["intelligence"]["doctor"].setdefault("meta", {})

    # 5) log
    data.setdefault("log", {"events": [], "analysis": []})
    data["log"].setdefault("events", [])
    data["log"].setdefault("analysis", [])

    # ✅ 6) ui (KeyError 방지 핵심)
    data.setdefault("ui", {})
    data["ui"].setdefault("patient_synced", False)
    data["ui"].setdefault("audit_log", [])
    data["ui"].setdefault("last_transcript_line", "")
    data["ui"].setdefault("ai_status", "idle")

    # 7) meta
    data.setdefault("meta", {})
    data["meta"].setdefault("created_ts", time.time())

    return data


# ============================================================
# 4) CentralHub wrapper (matches your reset/snapshot/log style)
# ============================================================

class CentralHub:
    """
    Convenience wrapper used across the app.

    - canonical state: session_state["hub_data"]
    - background queues: init_hub_queues(session_state)
    """

    def __init__(self, session_state: Any):
        self.session_state = session_state

        # simulation bookkeeping fields (optional)
        self.simulation_start_time: Optional[float] = None
        self.processed_event_indices: set[int] = set()

        # make sure queues exist
        init_hub_queues(self.session_state)

        # make sure hub_data exists
        if _unwrap_session(self.session_state).get("hub_data") is None:
            set_hub_data(self.session_state, default_hub_data())

    @property
    def data(self) -> Dict[str, Any]:
        return get_hub_data(self.session_state)

    @data.setter
    def data(self, new_data: Dict[str, Any]) -> None:
        set_hub_data(self.session_state, new_data)

    def reset(self) -> None:
        self.data = default_hub_data()
        self.simulation_start_time = None
        self.processed_event_indices = set()

    def snapshot(self) -> Dict[str, Any]:
        return copy.deepcopy(self.data)

    def log(self, message: str) -> None:
        """UI-friendly audit log."""
        ctx = ensure_schema(self.data)
        ctx["ui"]["audit_log"].append(message)
        ctx["ui"]["audit_log"] = ctx["ui"]["audit_log"][-200:]
        self.data = ctx

    def set_ai_status(self, status: str) -> None:
        """simulation_engine / intelligence_engine가 호출해도 안전하게."""
        ctx = ensure_schema(self.data)
        ctx["ui"]["ai_status"] = status
        self.data = ctx

    def append_event(self, event: Dict[str, Any]) -> None:
        ctx = ensure_schema(self.data)
        ctx.setdefault("log", {}).setdefault("events", []).append(event)
        self.data = ctx

    def apply_patient_snapshot(self, patient_snapshot: Dict[str, Any]) -> None:
        """
        EMR sync 같은 환자 스냅샷을 hub.data.patient에 merge.
        (없는 키는 건드리지 않고, 있는 키만 덮어씀)
        """
        if not patient_snapshot:
            return

        ctx = ensure_schema(self.data)
        p = ctx.setdefault("patient", {})

        for field in [
            "patient_id",
            "profile",
            "problems",
            "allergies",
            "medications",
            "history_summary",
            "historical_labs",
        ]:
            if field in patient_snapshot and patient_snapshot[field] is not None:
                p[field] = copy.deepcopy(patient_snapshot[field])

        # normalize sex key
        prof = p.setdefault("profile", {})
        if "gender" in prof and "sex" not in prof:
            prof["sex"] = prof.get("gender")

        ctx["patient"] = p
        self.data = ensure_schema(ctx)

    def apply_analysis_result(self, result: Dict[str, Any]) -> None:
        """
        IntelligenceEngine/Worker가 만든 결과(analysis result)를 hub.data에 안전하게 반영합니다.

        기대 입력 형태 예:
        {
          "triage": {"alerts": [...], "status": "...", "meta": {...}, "orders": [...]?},
          "doctor": {"note": "... or {note:{text}}", "suggested_orders": [...], "alerts": [...], ...},
          "timestamp": ...,
          "seq": ...,
          "reason": ...
        }
        """
        if not isinstance(result, dict):
            return

        ctx = ensure_schema(self.data)

        # --- triage ---
        triage = result.get("triage")
        if isinstance(triage, dict):
            tctx = ctx["intelligence"]["triage"]

            # alerts
            if isinstance(triage.get("alerts"), list):
                tctx["alerts"] = triage["alerts"]

            # status/meta
            if triage.get("status") is not None:
                tctx["status"] = triage.get("status")
            if isinstance(triage.get("meta"), dict):
                tctx["meta"] = triage["meta"]

            ctx["intelligence"]["triage"] = tctx

        # --- doctor ---
        doctor = result.get("doctor")
        if isinstance(doctor, dict):
            dctx = ctx["intelligence"]["doctor"]

            # note 처리: doctor["note"]가 str이거나 dict일 수 있음
            note_val = doctor.get("note")
            if isinstance(note_val, str):
                dctx["note"]["text"] = note_val
            elif isinstance(note_val, dict):
                # {"text": "..."} 또는 {"note":{"text":"..."}} 등 유연 처리
                if isinstance(note_val.get("text"), str):
                    dctx["note"]["text"] = note_val["text"]
                elif isinstance(note_val.get("note"), dict) and isinstance(note_val["note"].get("text"), str):
                    dctx["note"]["text"] = note_val["note"]["text"]

            # suggested_orders
            if isinstance(doctor.get("suggested_orders"), list):
                dctx["suggested_orders"] = doctor["suggested_orders"]

            # recommendations
            if isinstance(doctor.get("recommendations"), list):
                dctx["recommendations"] = doctor["recommendations"]

            # alerts
            if isinstance(doctor.get("alerts"), list):
                dctx["alerts"] = doctor["alerts"]

            # status/meta
            if doctor.get("status") is not None:
                dctx["status"] = doctor.get("status")
            if isinstance(doctor.get("meta"), dict):
                dctx["meta"] = doctor["meta"]

            ctx["intelligence"]["doctor"] = dctx

        # --- log / bookkeeping ---
        ctx.setdefault("log", {}).setdefault("analysis", []).append(
            {
                "ts": result.get("timestamp", time.time()),
                "seq": result.get("seq"),
                "reason": result.get("reason"),
                "has_triage": bool(result.get("triage")),
                "has_doctor": bool(result.get("doctor")),
                "error": result.get("error"),
            }
        )
        ctx["log"]["analysis"] = ctx["log"]["analysis"][-200:]

        # 상태 표시 (UI용)
        ctx["ui"]["ai_status"] = "done"

        self.data = ensure_schema(ctx)

    def set_patient_synced(self, synced: bool) -> None:
        ctx = ensure_schema(self.data)
        ctx["ui"]["patient_synced"] = bool(synced)
        self.data = ctx

    def set_last_transcript_line(self, text: str) -> None:
        ctx = ensure_schema(self.data)
        ctx["ui"]["last_transcript_line"] = text or ""
        self.data = ctx