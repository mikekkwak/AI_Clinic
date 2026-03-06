"""Simulation Engine

Scenario-driven demo runner.

Scenario JSON format (example):
{
  "scenario_name": "...",
  "patient_id": "P1001",
  "timeline": [
     {"seconds": 0, "type": "start", "data": {"vitals": {...}}},
     {"seconds": 5, "type": "transcript", "data": {"role": "patient", "text": "..."}},
  ]
}

This engine is intentionally simple.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Optional, Tuple

from .central_hub import CentralHub, ensure_schema
from .intelligence_engine import IntelligenceEngine, detect_immediate_symptom_and_flag


class SimulationEngine:
    def __init__(self, hub: CentralHub, brain: Optional[IntelligenceEngine] = None):
        self.hub = hub
        self.scenario_data: Optional[Dict[str, Any]] = None
        self.brain = brain

        project_root = os.path.dirname(os.path.dirname(__file__))
        self.base_path = os.path.join(project_root, "data", "scenarios")

    def _resolve_scenario_path(self, filename: str) -> Optional[str]:
        candidates = []
        if os.path.isabs(filename):
            candidates.append(filename)
        candidates.append(os.path.join(self.base_path, filename))
        candidates.append(os.path.join(os.getcwd(), filename))
        candidates.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), filename))

        for p in candidates:
            if p and os.path.exists(p):
                return p
        return None

    def load_scenario(self, filename: str) -> Tuple[bool, str]:
        path = self._resolve_scenario_path(filename)
        if not path:
            return False, f"Scenario file not found: {filename}"

        try:
            with open(path, "r", encoding="utf-8") as f:
                self.scenario_data = json.load(f)

            self.hub.reset()

            return True, "Loaded"
        except Exception as e:
            return False, str(e)

    def start(self) -> None:
        self.hub.simulation_start_time = time.time()
        self.hub.processed_event_indices = set()

    def update(self) -> bool:
        if self.hub.simulation_start_time is None or self.scenario_data is None:
            return False

        elapsed = time.time() - self.hub.simulation_start_time
        timeline = self.scenario_data.get("timeline", [])
        changed = False

        for idx, event in enumerate(timeline):
            if idx in self.hub.processed_event_indices:
                continue
            if elapsed >= float(event.get("seconds", 0)):
                self._apply_event(event)
                self.hub.processed_event_indices.add(idx)
                changed = True

        return changed

    def _apply_event(self, event: Dict[str, Any]) -> None:
        etype = event.get("type")
        data = event.get("data") or {}

        ctx = ensure_schema(self.hub.data)

        # log
        ctx.setdefault("log", {}).setdefault("events", []).append({"ts": time.time(), "source": "simulation", **event})

        if etype == "start":
            vitals = data.get("vitals") or {}

            # ✅ Sync 전에는 vitals를 visit.vitals.latest에 넣지 않는다
            # 대신 meta.hidden_start_vitals에 숨겨둔다
            ctx.setdefault("meta", {})
            ctx["meta"]["hidden_start_vitals"] = vitals

        elif etype == "transcript":
            msg = {"role": data.get("role"), "text": data.get("text"), "ts": time.time()}
            ctx["visit"].setdefault("transcript", []).append(msg)

            # immediate keyword alert
            immediate_alerts = detect_immediate_symptom_and_flag(ctx["visit"]["transcript"])
            if immediate_alerts:
                ctx["intelligence"]["triage"].setdefault("alerts", [])
                ctx["intelligence"]["triage"]["alerts"] = immediate_alerts + (ctx["intelligence"]["triage"].get("alerts") or [])

            self.hub.data = ctx

            # enqueue full analysis
            if self.brain is not None:
                self.hub.set_ai_status("thinking")
                self.brain.enqueue_from_hub(self.hub, reason="transcript")
            return

        elif etype == "lab_result":
            ctx["visit"].setdefault("labs", {"latest": [], "critical": []})
            ctx["visit"]["labs"]["latest"] = data

        self.hub.data = ctx
