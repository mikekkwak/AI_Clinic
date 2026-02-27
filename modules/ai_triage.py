"""AI Triage module

- Rule 기반 감시(즉시성): vitals threshold + 고위험 키워드/패턴
- (옵션) LLM 기반 감시(민감도 향상): CTAS 추정 + reasoning

출력은 항상 UI-friendly dict 형태:
{
  "alerts": [ {"title": str, "criteria": str, "severity": "high|medium|low"} ... ],
  "orders": [ ... optional ... ],
  "meta": { ... }
}

NOTE: 의료기기/임상 의사결정 자동화가 아니라, clinician-in-the-loop 의사결정 보조를 전제로 합니다.
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore


load_dotenv()


def _safe_float(x: Any) -> Optional[float]:
    """Parse numbers from int/float/str like '97%', '98.4F', ' 88 '."""
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return None
        m = re.search(r"-?\d+(?:\.\d+)?", s)
        if not m:
            return None
        try:
            return float(m.group(0))
        except Exception:
            return None
    return None


def _norm_text(s: str) -> str:
    return (s or "").lower().strip()


def _join_patient_utterances(transcript: List[Dict[str, Any]]) -> str:
    texts: List[str] = []
    for msg in transcript or []:
        if isinstance(msg, dict) and msg.get("role") == "patient":
            t = msg.get("text")
            if isinstance(t, str) and t.strip():
                texts.append(t.strip())
    return " ".join(texts)


@dataclass
class TriageThresholds:
    HR_low_critical: float = 40
    HR_low_warning: float = 50
    HR_high_warning: float = 120
    HR_high_critical: float = 150

    SpO2_critical: float = 90
    SpO2_warning: float = 94

    SBP_critical_low: float = 85
    SBP_warning_low: float = 95
    SBP_warning_high: float = 160
    SBP_critical_high: float = 200

    DBP_critical_low: float = 50
    DBP_warning_low: float = 60
    DBP_warning_high: float = 90
    DBP_critical_high: float = 120

    TempF_critical_low: float = 90
    TempF_warning_low: float = 95
    TempF_warning_high: float = 100.4
    TempF_critical_high: float = 104

    RR_critical_low: float = 6
    RR_warning_low: float = 8
    RR_warning_high: float = 25
    RR_critical_high: float = 30


class AITriage:
    def __init__(self, *, model: Optional[str] = None):
        self.model = model or os.getenv("OPENAI_MODEL_TRIAGE", "gpt-4o")

        api_key = os.getenv("OPENAI_API_KEY")
        if api_key and OpenAI is not None:
            self.client = OpenAI(api_key=api_key)
        else:
            self.client = None

        self.th = TriageThresholds()

        # 빠른 키워드 감시(영/한)
        self.urgent_keywords = {
            # Korean
            "식은땀",
            "가슴 통증",
            "호흡곤란",
            "숨이 차",
            "어깨",
            "가슴이 답답",
            "심한 통증",
            "실신",
            # English
            "chest pain",
            "chest pressure",
            "short of breath",
            "shortness of breath",
            "dyspnea",
            "clammy",
            "cold sweat",
            "sweating",
            "syncope",
            "left shoulder",
            "jaw",
            "arm pain",
            "back pain",
        }

    # -----------------------------
    # Vitals evaluation
    # -----------------------------

    def evaluate_vitals(self, vitals: Dict[str, Any]) -> Tuple[int, List[str]]:
        urgency_level = 5
        reasons: List[str] = []

        hr = _safe_float(vitals.get("hr"))
        if hr is not None:
            if hr < self.th.HR_low_critical or hr > self.th.HR_high_critical:
                urgency_level = min(urgency_level, 1)
                reasons.append(f"Heart rate is critically abnormal ({hr:.0f} bpm).")
            elif hr < self.th.HR_low_warning or (self.th.HR_high_warning < hr <= self.th.HR_high_critical):
                urgency_level = min(urgency_level, 2)
                reasons.append(f"Heart rate is concerning ({hr:.0f} bpm).")

        spo2 = _safe_float(vitals.get("spo2"))
        if spo2 is not None:
            if spo2 <= self.th.SpO2_critical:
                urgency_level = min(urgency_level, 1)
                reasons.append(f"Oxygen saturation is critically low ({spo2:.0f}%).")
            elif spo2 <= self.th.SpO2_warning:
                urgency_level = min(urgency_level, 2)
                reasons.append(f"Oxygen saturation is low ({spo2:.0f}%).")

        bp = vitals.get("bp")
        if isinstance(bp, str) and re.match(r"^\s*\d+\s*/\s*\d+\s*$", bp):
            systolic, diastolic = [int(x) for x in bp.split("/")]

            if systolic < self.th.SBP_critical_low or systolic > self.th.SBP_critical_high:
                urgency_level = min(urgency_level, 1)
                reasons.append(f"Systolic BP is critically abnormal ({systolic} mmHg).")
            elif systolic < self.th.SBP_warning_low or systolic > self.th.SBP_warning_high:
                urgency_level = min(urgency_level, 2)
                reasons.append(f"Systolic BP is concerning ({systolic} mmHg).")

            if diastolic < self.th.DBP_critical_low or diastolic > self.th.DBP_critical_high:
                urgency_level = min(urgency_level, 1)
                reasons.append(f"Diastolic BP is critically abnormal ({diastolic} mmHg).")
            elif diastolic < self.th.DBP_warning_low or diastolic > self.th.DBP_warning_high:
                urgency_level = min(urgency_level, 2)
                reasons.append(f"Diastolic BP is concerning ({diastolic} mmHg).")

        rr = _safe_float(vitals.get("rr"))
        if rr is not None:
            if rr < self.th.RR_critical_low or rr > self.th.RR_critical_high:
                urgency_level = min(urgency_level, 1)
                reasons.append(f"Respiratory rate is critically abnormal ({rr:.0f} rpm).")
            elif rr < self.th.RR_warning_low or rr > self.th.RR_warning_high:
                urgency_level = min(urgency_level, 2)
                reasons.append(f"Respiratory rate is concerning ({rr:.0f} rpm).")

        temp = _safe_float(vitals.get("temp"))
        if temp is not None:
            if temp < self.th.TempF_critical_low or temp > self.th.TempF_critical_high:
                urgency_level = min(urgency_level, 1)
                reasons.append(f"Temperature is critically abnormal ({temp:.1f}°F).")
            elif temp < self.th.TempF_warning_low or temp > self.th.TempF_warning_high:
                urgency_level = min(urgency_level, 2)
                reasons.append(f"Temperature is concerning ({temp:.1f}°F).")

        return urgency_level, reasons

    # -----------------------------
    # Symptom rules
    # -----------------------------

    def _has_any_keyword(self, text: str) -> List[str]:
        t = _norm_text(text)
        hits = []
        for kw in self.urgent_keywords:
            if kw in t:
                hits.append(kw)
        return hits

    def evaluate_symptom_rules(self, hub_data: Dict[str, Any]) -> Tuple[int, List[Dict[str, Any]]]:
        ctx = hub_data or {}
        p = (ctx.get("patient") or {})
        prof = (p.get("profile") or {})
        age = _safe_float(prof.get("age"))
        problems = p.get("problems") or []
        problems_norm = " ".join([str(x).lower() for x in problems])

        transcript = (ctx.get("visit") or {}).get("transcript") or []
        patient_text = _join_patient_utterances(transcript)
        t = _norm_text(patient_text)

        alerts: List[Dict[str, Any]] = []
        level = 5

        kw_hits = self._has_any_keyword(patient_text)

        # simple ACS heuristic (demo)
        has_epigastric = any(x in t for x in ["epigastr", "upper stomach", "upper abdomen", "indigestion", "heartburn", "stomach"])  # broad
        has_chest = any(x in t for x in ["chest", "pressure", "tight", "heav", "squeez", "crush"])  # broad
        has_radiation = any(x in t for x in ["left shoulder", "arm", "jaw", "back", "neck"])
        has_diaphoresis = any(x in t for x in ["cold sweat", "sweat", "clammy", "diaphor"]) or ("식은땀" in t)
        has_dyspnea = any(x in t for x in ["short of breath", "shortness of breath", "winded", "dyspnea", "숨이", "호흡곤란"])
        has_nausea = any(x in t for x in ["nausea", "vomit", "구역", "토"])

        risk_factors = []
        if "dm" in problems_norm or "diabetes" in problems_norm:
            risk_factors.append("DM")
        if "htn" in problems_norm or "hypertension" in problems_norm:
            risk_factors.append("HTN")
        if "hyperlip" in problems_norm or "dyslip" in problems_norm:
            risk_factors.append("HLD")

        high_risk_context = False
        if age is not None and age >= 60:
            high_risk_context = True
        if risk_factors:
            high_risk_context = True

        if (has_chest or has_epigastric) and (has_diaphoresis or has_dyspnea or has_radiation or has_nausea) and high_risk_context:
            level = min(level, 2)
            alerts.append(
                {
                    "severity": "high",
                    "title": "Warning: Possible ACS / MI (Atypical Presentation)",
                    "criteria": "High-risk context (age/risk factors) with epigastric/chest discomfort plus associated symptoms (e.g., diaphoresis, dyspnea, nausea, or radiation).",
                    "recommended_actions": [
                        "Obtain 12-lead ECG ASAP (goal <10 min)",
                        "Order serial troponins",
                        "Place on cardiac monitor",
                    ],
                }
            )

        if has_chest and (has_diaphoresis or "syncope" in t or "실신" in t) and high_risk_context:
            level = min(level, 1)

        if kw_hits and level > 2:
            level = min(level, 2)
            alerts.append(
                {
                    "severity": "medium",
                    "title": "Warning: High-risk symptom keywords detected",
                    "criteria": " / ".join(sorted(set(kw_hits)))[:220],
                }
            )

        return level, alerts

    # -----------------------------
    # LLM-based CTAS (optional)
    # -----------------------------

    def evaluate_symptoms_llm(self, patient_history: str, current_symptoms: str) -> Tuple[int, str]:
        if not self.client:
            return 5, "OPENAI_API_KEY missing (LLM triage skipped)."

        if not (current_symptoms or "").strip():
            return 5, "No recent symptoms reported."

        prompt = f"""
You are an expert ER triage assistant.

Patient PMHx: {patient_history}
Current Symptoms: {current_symptoms}

Task:
1) Estimate CTAS level (1 to 5).
2) Explain reasoning in 3-6 bullet points.

Respond strictly in this format:
CTAS Level: [1-5]
Reasons: - ...\n- ...
""".strip()

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a medical assistant trained in CTAS triage."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=240,
                temperature=0.0,
            )
            txt = (resp.choices[0].message.content or "").strip()

            ctas_level = 5
            reasons = "Unknown"
            m = re.search(r"CTAS\s*Level\s*:\s*(\d)", txt)
            if m:
                ctas_level = int(m.group(1))
            if "Reasons:" in txt:
                reasons = txt.split("Reasons:", 1)[1].strip()

            return ctas_level, reasons
        except Exception as e:  # pragma: no cover
            return 5, f"LLM Evaluation Error: {e}"

    # -----------------------------
    # Public API
    # -----------------------------

    def analyze(self, hub_data: Dict[str, Any], *, reason: str = "periodic") -> Dict[str, Any]:
        hub_data = hub_data or {}
        visit = hub_data.get("visit") or {}

        alerts: List[Dict[str, Any]] = []
        orders: List[Dict[str, Any]] = []
        meta: Dict[str, Any] = {"reason": reason, "ts": time.time(), "llm_used": False}

        # 1) vitals
        vitals_latest = ((visit.get("vitals") or {}).get("latest") or {})
        v_level, v_reasons = self.evaluate_vitals(vitals_latest)
        if v_level <= 2 and v_reasons:
            title = "CRITICAL: Vitals Instability" if v_level == 1 else "Warning: Abnormal Vitals"
            alerts.append({"severity": "high" if v_level == 1 else "medium", "title": title, "criteria": " / ".join(v_reasons)})

        # 2) rule-based symptoms
        s_level_rules, rule_alerts = self.evaluate_symptom_rules(hub_data)
        if rule_alerts:
            alerts.extend(rule_alerts)

        # 3) optional LLM
        transcript = visit.get("transcript") or []
        patient_text = _join_patient_utterances(transcript)
        if patient_text.strip() and self.client:
            problems = (hub_data.get("patient") or {}).get("problems") or []
            pmhx = ", ".join([str(x) for x in problems]) if problems else "None"

            ctas_level, ctas_reasons = self.evaluate_symptoms_llm(pmhx, patient_text)
            meta["llm_used"] = True
            meta["ctas"] = ctas_level
            if ctas_level <= 2:
                title = f"CRITICAL: High Risk Symptoms (CTAS {ctas_level})" if ctas_level == 1 else f"Warning: Concerning Symptoms (CTAS {ctas_level})"
                alerts.append({"severity": "high" if ctas_level == 1 else "medium", "title": title, "criteria": ctas_reasons})

        # dedup
        seen = set()
        deduped = []
        for a in alerts:
            key = (a.get("title"), a.get("criteria"))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(a)

        # minimal triage order set
        if any("ACS" in (a.get("title") or "") or "MI" in (a.get("title") or "") for a in deduped):
            orders.append({"title": "Cardiac (Triage)", "items": ["12-lead ECG", "Troponin (serial)", "Cardiac monitor", "IV access"], "priority": "STAT"})

        meta["vitals_urgency"] = v_level
        meta["symptom_rule_level"] = s_level_rules

        return {"alerts": deduped, "orders": orders, "meta": meta}
