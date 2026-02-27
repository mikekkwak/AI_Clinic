"""SmartLab (demo)

- 외부에서 labs를 가져오는 것처럼 연출
- hub.schema에 맞게 patient.historical_labs + visit.labs.latest / critical 업데이트

실제 구현에서는 LIS/EMR 인터페이스 결과를 event로 받아서 CentralHub에 넣는 역할.
"""

from __future__ import annotations

import time
from typing import Optional

import plotly.graph_objects as go

from .central_hub import ensure_schema


LAB_CONFIG = {
    "Troponin-I": {"unit": "ng/mL", "ref_max": 0.04},
    "Creatinine": {"unit": "mg/dL", "ref_max": 1.2},
    "WBC": {"unit": "x10^3/uL", "ref_max": 11.0},
}


class SmartLab:
    def __init__(self, hub):
        self.hub = hub

    def simulate_external_import(self) -> bool:
        time.sleep(0.6)

        imported_history = {
            "Troponin-I": [
                {"dt": "2023-01-15 09:00", "value": 0.01, "unit": "ng/mL", "ref_range": "< 0.04"},
                {"dt": "2023-06-20 09:00", "value": 0.02, "unit": "ng/mL", "ref_range": "< 0.04"},
            ],
            "Creatinine": [
                {"dt": "2023-01-15 09:00", "value": 1.5, "unit": "mg/dL", "ref_range": "0.6 - 1.2"},
            ],
        }

        current_labs = [
            {"name": "Troponin-I", "value": 0.45, "unit": "ng/mL", "flag": "High", "dt": "2026-02-12 10:15"},
            {"name": "Creatinine", "value": 1.6, "unit": "mg/dL", "flag": "High", "dt": "2026-02-12 10:15"},
            {"name": "WBC", "value": 8.5, "unit": "x10^3/uL", "flag": "Normal", "dt": "2026-02-12 10:15"},
        ]

        ctx = ensure_schema(self.hub.data)

        # merge history
        h = ctx["patient"].setdefault("historical_labs", {})
        for lab, rows in imported_history.items():
            h.setdefault(lab, [])
            h[lab].extend(rows)

        # append current for trend
        for row in current_labs:
            lab = row["name"]
            h.setdefault(lab, [])
            h[lab].append(
                {
                    "dt": row["dt"],
                    "value": row["value"],
                    "unit": row["unit"],
                    "ref_range": f"< {LAB_CONFIG.get(lab, {}).get('ref_max', '')}" if LAB_CONFIG.get(lab) else "",
                }
            )

        # visit
        ctx["visit"].setdefault("labs", {"latest": [], "critical": []})
        ctx["visit"]["labs"]["latest"] = current_labs

        critical = []
        for row in current_labs:
            lab = row["name"]
            ref = LAB_CONFIG.get(lab)
            if ref and isinstance(row.get("value"), (int, float)) and row["value"] > ref["ref_max"]:
                critical.append(row)
        ctx["visit"]["labs"]["critical"] = critical

        self.hub.data = ctx
        return True

    def get_trend_graph(self, lab_name: str) -> Optional[go.Figure]:
        ctx = ensure_schema(self.hub.data)
        hist = (ctx["patient"].get("historical_labs") or {}).get(lab_name) or []
        if not hist:
            return None

        dates = [d.get("dt") for d in hist]
        values = [d.get("value") for d in hist]

        fig = go.Figure()
        ref = LAB_CONFIG.get(lab_name)
        if ref:
            fig.add_hrect(y0=0, y1=ref["ref_max"], fillcolor="green", opacity=0.1, line_width=0)

        fig.add_trace(go.Scatter(x=dates, y=values, mode="lines+markers", name=lab_name))
        fig.update_layout(title=f"{lab_name} Trend", height=260, margin=dict(l=20, r=20, t=45, b=20))
        return fig
