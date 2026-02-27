"""
Intelligence Engine (Safe, Session-State Native)

- Background worker (thread) that runs AI Triage + AI Doctor on a hub snapshot.
- Queue-based so Streamlit UI stays responsive.
- Stores queues + thread handles directly inside st.session_state (no fragile central_hub dependency).
- Provides immediate keyword detection for instant alerts.
"""

from __future__ import annotations

import queue
import threading
import time
import traceback
from typing import Any, Dict, List, Optional

from .ai_doctor import AIDoctor
from .ai_triage import AITriage


# -----------------------------
# Immediate keyword detection
# -----------------------------

URGENT_KEYWORDS = {
    # Korean
    "식은땀",
    "가슴 통증",
    "호흡곤란",
    "숨이 차",
    "실신",
    "의식 저하",
    "피 토",
    "검은 변",
    # English
    "chest pain",
    "chest pressure",
    "shortness of breath",
    "short of breath",
    "syncope",
    "faint",
    "clammy",
    "cold sweat",
    "severe pain",
    "blood in vomit",
    "black stool",
}


def _extract_text_from_latest(transcript_snapshot: Any) -> str:
    """
    Accepts:
    - list[dict] transcript entries {role,text}
    - list[str]
    - str
    Returns best-effort latest utterance text.
    """
    if transcript_snapshot is None:
        return ""
    if isinstance(transcript_snapshot, str):
        return transcript_snapshot
    if isinstance(transcript_snapshot, list) and transcript_snapshot:
        latest = transcript_snapshot[-1]
        if isinstance(latest, str):
            return latest
        if isinstance(latest, dict):
            return str(latest.get("text") or "")
    return ""


def detect_immediate_symptom_and_flag(transcript_snapshot: Any) -> List[Dict[str, Any]]:
    alerts: List[Dict[str, Any]] = []
    txt = (_extract_text_from_latest(transcript_snapshot) or "").lower().strip()
    if not txt:
        return alerts

    hits = [kw for kw in URGENT_KEYWORDS if kw in txt]
    if hits:
        alerts.append(
            {
                "severity": "high",
                "title": "Immediate: High-risk symptom keyword detected",
                "criteria": " / ".join(sorted(set(hits)))[:220],
                "recommended_actions": [
                    "Reassess patient immediately",
                    "Obtain vitals",
                    "Consider ECG + oxygen + monitor if cardiopulmonary concern",
                ],
                "timestamp": time.time(),
            }
        )
    return alerts


# -----------------------------
# Background worker
# -----------------------------

class AnalysisWorker(threading.Thread):
    def __init__(self, task_q: "queue.Queue", result_q: "queue.Queue", stop_event: threading.Event):
        super().__init__(daemon=True)
        self.task_q = task_q
        self.result_q = result_q
        self.stop_event = stop_event

        # Create analyzers once per thread
        self.triage = AITriage()
        self.doctor = AIDoctor()

    def run(self):
        while not self.stop_event.is_set():
            try:
                task = self.task_q.get(timeout=0.5)
            except Exception:
                continue

            try:
                reason = task.get("reason", "periodic")
                seq = task.get("seq")
                hub_snapshot = task.get("hub_snapshot")

                # Backward compatibility: allow tasks with only transcript
                if hub_snapshot is None:
                    transcript = task.get("transcript", [])
                    hub_snapshot = {
                        "patient": {},
                        "visit": {"transcript": transcript, "vitals": {"latest": {}}},
                        "intelligence": {},
                        "log": {},
                    }

                # Run triage + doctor
                try:
                    triage_out = self.triage.analyze(hub_snapshot, reason=reason)
                except Exception:
                    triage_out = {
                        "alerts": [],
                        "orders": [],
                        "meta": {"error": "triage failed", "trace": traceback.format_exc()},
                    }

                try:
                    doctor_out = self.doctor.analyze(hub_snapshot, reason=reason)
                except Exception:
                    doctor_out = {
                        "note": "",
                        "suggested_orders": [],
                        "recommendations": [],
                        "alerts": [],
                        "meta": {"error": "doctor failed", "trace": traceback.format_exc()},
                    }

                merged = {
                    "triage": triage_out,
                    "doctor": doctor_out,
                    "timestamp": time.time(),
                    "seq": seq,
                    "reason": reason,
                }
                self.result_q.put(merged)

            except Exception as e:
                self.result_q.put(
                    {
                        "error": str(e),
                        "trace": traceback.format_exc(),
                        "timestamp": time.time(),
                        "seq": task.get("seq"),
                        "reason": task.get("reason"),
                    }
                )
            finally:
                try:
                    self.task_q.task_done()
                except Exception:
                    pass


# -----------------------------
# Public engine wrapper
# -----------------------------

class IntelligenceEngine:
    """
    Per-session wrapper.

    Pass st.session_state (Streamlit SessionStateProxy) OR any dict-like object.
    This implementation does NOT depend on central_hub internals.
    """

    # session keys (namespaced to avoid collisions)
    _K_TASK_Q = "_ai_task_queue"
    _K_RESULT_Q = "_ai_result_queue"
    _K_STOP = "_ai_stop_event"
    _K_THREAD = "_ai_worker_thread"

    def __init__(self, session_state: Optional[Any] = None):
        self.session_state = session_state

    def init_with_session(self, session_state: Any) -> "IntelligenceEngine":
        self.session_state = session_state
        return self

    def _ss(self):
        """
        Streamlit session_state is dict-like.
        We avoid isinstance(..., dict) checks on purpose.
        """
        if self.session_state is None:
            raise RuntimeError("IntelligenceEngine: session_state is None. Pass st.session_state.")
        return self.session_state

    def _ensure_queues(self) -> Dict[str, Any]:
        ss = self._ss()

        if self._K_TASK_Q not in ss or ss.get(self._K_TASK_Q) is None:
            ss[self._K_TASK_Q] = queue.Queue()

        if self._K_RESULT_Q not in ss or ss.get(self._K_RESULT_Q) is None:
            ss[self._K_RESULT_Q] = queue.Queue()

        if self._K_STOP not in ss or ss.get(self._K_STOP) is None:
            ss[self._K_STOP] = threading.Event()

        return {
            "task_queue": ss[self._K_TASK_Q],
            "result_queue": ss[self._K_RESULT_Q],
            "stop_event": ss[self._K_STOP],
        }

    def start(self) -> "IntelligenceEngine":
        ss = self._ss()
        hub_objs = self._ensure_queues()

        # thread already exists?
        if self._K_THREAD in ss and ss.get(self._K_THREAD) is not None:
            return self

        worker = AnalysisWorker(hub_objs["task_queue"], hub_objs["result_queue"], hub_objs["stop_event"])
        worker.start()
        ss[self._K_THREAD] = worker
        return self

    def enqueue(self, hub_snapshot: Dict[str, Any], *, seq: Optional[int] = None, reason: str = "periodic") -> None:
        hub_objs = self._ensure_queues()
        hub_objs["task_queue"].put({"hub_snapshot": hub_snapshot, "seq": seq, "reason": reason})

    def enqueue_from_hub(self, hub: Any, *, reason: str = "periodic") -> None:
        """
        Supports:
        - hub.snapshot() if available
        - hub.data if available
        - fallback: hub itself if already dict-like
        """
        snapshot = None

        try:
            snapshot = hub.snapshot()
        except Exception:
            pass

        if snapshot is None:
            try:
                snapshot = hub.data  # CentralHub-style
            except Exception:
                pass

        if snapshot is None and isinstance(hub, dict):
            snapshot = hub

        if snapshot is None:
            snapshot = {"patient": {}, "visit": {"transcript": [], "vitals": {"latest": {}}}}

        seq_val = None
        try:
            seq_val = len(((snapshot.get("visit") or {}).get("transcript") or []))
        except Exception:
            seq_val = None

        self.enqueue(snapshot, seq=seq_val, reason=reason)

    def drain_results(self, *, limit: int = 50) -> List[Dict[str, Any]]:
        hub_objs = self._ensure_queues()
        result_q = hub_objs["result_queue"]

        out: List[Dict[str, Any]] = []
        for _ in range(limit):
            try:
                res = result_q.get_nowait()
            except Exception:
                break
            out.append(res)
        return out

    def stop(self) -> None:
        ss = self._ss()
        hub_objs = self._ensure_queues()
        hub_objs["stop_event"].set()

        # Optionally clear the thread handle so we can restart later
        try:
            ss[self._K_THREAD] = None
        except Exception:
            pass