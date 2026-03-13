"""AI Triage module

Rule-based triage alerting only.

Goals:
- Produce alerts only (no orders)
- Detect immediate high-risk states
- Collapse multiple signals into a single triage summary alert
- Support alert replacement when new vitals/labs/transcript data arrive
- Keep structure extensible for future LLM integration

Output is always a UI-friendly dict:
{
  "alerts": [
    {
      "id": str,
      "severity": "high|medium|low",
      "title": str,
      "criteria": str,
      "urgency": str,
      "timestamp": float
    }
  ],
  "status": "done",
  "meta": { ... }
}

NOTE:
This module is intended for clinician-in-the-loop decision support only.
It is not intended to function as a standalone medical device or
fully autonomous clinical decision-maker.
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

try:
    from dotenv import load_dotenv
except Exception:
    def load_dotenv(*args, **kwargs):
        return False

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore


load_dotenv()


def _safe_float(x: Any) -> Optional[float]:
    """Parse a numeric value from int/float/str like '97%', '98.4F', ' 88 '."""
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


def _parse_bp(bp: Any) -> Tuple[Optional[int], Optional[int]]:
    """Return (SBP, DBP) if BP string looks like '120/80'."""
    if not isinstance(bp, str):
        return None, None
    m = re.match(r"^\s*(\d+)\s*/\s*(\d+)\s*$", bp)
    if not m:
        return None, None
    try:
        return int(m.group(1)), int(m.group(2))
    except Exception:
        return None, None


def _short_join(items: List[str], limit: int = 5) -> str:
    cleaned = [str(x).strip() for x in items if str(x).strip()]
    if not cleaned:
        return ""
    if len(cleaned) <= limit:
        return ", ".join(cleaned)
    return ", ".join(cleaned[:limit]) + f", +{len(cleaned) - limit} more"


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
    SBP_warning_high: float = 180
    SBP_critical_high: float = 200

    DBP_critical_low: float = 50
    DBP_warning_low: float = 60
    DBP_warning_high: float = 110
    DBP_critical_high: float = 120

    TempF_critical_low: float = 90
    TempF_warning_low: float = 95
    TempF_warning_high: float = 100.4
    TempF_critical_high: float = 104

    RR_critical_low: float = 6
    RR_warning_low: float = 8
    RR_warning_high: float = 25
    RR_critical_high: float = 30

    # SIRS thresholds
    SIRS_temp_high_f: float = 100.4
    SIRS_temp_low_f: float = 96.8
    SIRS_hr: float = 90
    SIRS_rr: float = 20

    # Sepsis / instability modifiers
    Sepsis_SBP_concern: float = 90
    Sepsis_SpO2_concern: float = 92

    # qSOFA-like modifier
    qSOFA_rr: float = 22
    qSOFA_sbp: float = 100


class AITriage:
    def __init__(self, *, model: Optional[str] = None, enable_llm: bool = True):
        self.model = model or os.getenv("OPENAI_MODEL_TRIAGE", "gpt-4o")
        self.enable_llm = enable_llm

        api_key = os.getenv("OPENAI_API_KEY")
        if api_key and OpenAI is not None and self.enable_llm:
            self.client = OpenAI(api_key=api_key)
        else:
            self.client = None

        self.th = TriageThresholds()

        self.urgent_keywords = {
            "식은땀",
            "가슴 통증",
            "호흡곤란",
            "숨이 차",
            "어깨",
            "가슴이 답답",
            "심한 통증",
            "실신",
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

        self.infection_keywords = {
            "fever",
            "febrile",
            "chills",
            "rigors",
            "infection",
            "sepsis",
            "cough",
            "phlegm",
            "sputum",
            "pneumonia",
            "uti",
            "urinary tract infection",
            "burning urination",
            "dysuria",
            "flank pain",
            "cellulitis",
            "wound infection",
            "abscess",
            "diarrhea",
            "vomiting",
            "abdominal infection",
            "appendicitis",
            "pyelonephritis",
            "rlq pain",
            "right lower quadrant",
            "구토",
            "설사",
            "열",
            "발열",
            "오한",
            "기침",
            "가래",
            "폐렴",
            "요로감염",
            "배뇨통",
            "옆구리 통증",
            "봉와직염",
            "상처 감염",
            "농양",
            "우하복부",
            "충수염",
        }

        self.mental_status_keywords = {
            "confused",
            "confusion",
            "disoriented",
            "difficult to wake",
            "lethargic",
            "altered mental status",
            "not acting right",
            "혼돈",
            "의식 저하",
            "멍하다",
            "헛소리",
            "정신이 없다",
        }

    # -----------------------------
    # Helpers
    # -----------------------------

    def _build_alert(
        self,
        *,
        alert_id: str,
        severity: str,
        title: str,
        criteria: str,
        urgency: str,
        source: str,
        extras: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        alert = {
            "id": alert_id,
            "severity": severity,
            "title": title,
            "criteria": criteria,
            "urgency": urgency,
            "timestamp": time.time(),
            "source": source,
        }
        if extras:
            alert.update(extras)
        return alert

    def _has_any_keyword(self, text: str, keywords: set[str]) -> List[str]:
        t = _norm_text(text)
        hits: List[str] = []
        for kw in keywords:
            if kw in t:
                hits.append(kw)
        return sorted(set(hits))

    def _extract_problems_text(self, hub_data: Dict[str, Any]) -> str:
        patient = hub_data.get("patient") or {}
        problems = patient.get("problems") or patient.get("pmhx") or []
        if isinstance(problems, list):
            return " ".join(str(x).lower() for x in problems)
        return str(problems).lower()

    def _extract_current_labs(self, hub_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        patient = hub_data.get("patient") or {}
        labs = patient.get("current_labs") or []
        return labs if isinstance(labs, list) else []

    def _find_lab_value(self, labs: List[Dict[str, Any]], test_names: List[str]) -> Optional[float]:
        aliases = {x.lower() for x in test_names}
        for row in labs:
            if not isinstance(row, dict):
                continue
            name = str(row.get("test") or row.get("name") or row.get("lab") or "").lower()
            if name in aliases:
                for key in ("value", "result", "current", "current_result"):
                    val = _safe_float(row.get(key))
                    if val is not None:
                        return val
                val = _safe_float(row.get("flag_value"))
                if val is not None:
                    return val
        return None

    # -----------------------------
    # Signal extraction
    # -----------------------------

    def _collect_vital_signals(self, vitals: Dict[str, Any]) -> Dict[str, Any]:
        signals: List[str] = []
        severe_signals: List[str] = []
        level = 5

        hr = _safe_float(vitals.get("hr"))
        rr = _safe_float(vitals.get("rr"))
        temp = _safe_float(vitals.get("temp"))
        spo2 = _safe_float(vitals.get("spo2"))
        sbp, dbp = _parse_bp(vitals.get("bp"))

        if hr is not None:
            if hr < self.th.HR_low_critical or hr > self.th.HR_high_critical:
                level = min(level, 1)
                severe_signals.append(f"critical heart rate ({hr:.0f} bpm)")
            elif hr < self.th.HR_low_warning or hr > self.th.HR_high_warning:
                level = min(level, 2)
                signals.append(f"tachycardia/bradycardia concern (HR {hr:.0f})")

        if spo2 is not None:
            if spo2 <= self.th.SpO2_critical:
                level = min(level, 1)
                severe_signals.append(f"critical hypoxemia (SpO2 {spo2:.0f}%)")
            elif spo2 <= self.th.SpO2_warning:
                level = min(level, 2)
                signals.append(f"low oxygen saturation (SpO2 {spo2:.0f}%)")

        if sbp is not None:
            if sbp < self.th.SBP_critical_low or sbp > self.th.SBP_critical_high:
                level = min(level, 1)
                severe_signals.append(f"critical systolic blood pressure ({sbp} mmHg)")
            elif sbp < self.th.SBP_warning_low or sbp > self.th.SBP_warning_high:
                level = min(level, 2)
                signals.append(f"systolic blood pressure concern (SBP {sbp})")

        if dbp is not None:
            if dbp < self.th.DBP_critical_low or dbp > self.th.DBP_critical_high:
                level = min(level, 1)
                severe_signals.append(f"critical diastolic blood pressure ({dbp} mmHg)")
            elif dbp < self.th.DBP_warning_low or dbp > self.th.DBP_warning_high:
                level = min(level, 2)
                signals.append(f"diastolic blood pressure concern (DBP {dbp})")

        if rr is not None:
            if rr < self.th.RR_critical_low or rr > self.th.RR_critical_high:
                level = min(level, 1)
                severe_signals.append(f"critical respiratory rate ({rr:.0f}/min)")
            elif rr < self.th.RR_warning_low or rr > self.th.RR_warning_high:
                level = min(level, 2)
                signals.append(f"respiratory rate concern (RR {rr:.0f})")

        if temp is not None:
            if temp < self.th.TempF_critical_low or temp > self.th.TempF_critical_high:
                level = min(level, 1)
                severe_signals.append(f"critical temperature abnormality ({temp:.1f}°F)")
            elif temp < self.th.TempF_warning_low or temp > self.th.TempF_warning_high:
                level = min(level, 2)
                signals.append(f"fever/hypothermia concern (Temp {temp:.1f}°F)")

        return {
            "level": level,
            "signals": signals,
            "severe_signals": severe_signals,
            "hr": hr,
            "rr": rr,
            "temp": temp,
            "spo2": spo2,
            "sbp": sbp,
            "dbp": dbp,
        }

    def _collect_acs_signals(self, hub_data: Dict[str, Any]) -> Dict[str, Any]:
        patient = hub_data.get("patient") or {}
        profile = patient.get("profile") or {}
        age = _safe_float(profile.get("age"))
        problems_norm = self._extract_problems_text(hub_data)

        transcript = (hub_data.get("visit") or {}).get("transcript") or []
        patient_text = _join_patient_utterances(transcript)
        t = _norm_text(patient_text)

        has_epigastric = any(
            x in t for x in ["epigastr", "upper stomach", "upper abdomen", "indigestion", "heartburn", "stomach"]
        )
        has_chest = any(
            x in t for x in ["chest", "pressure", "tight", "heav", "squeez", "crush"]
        )
        has_radiation = any(
            x in t for x in ["left shoulder", "arm", "jaw", "back", "neck"]
        )
        has_diaphoresis = any(
            x in t for x in ["cold sweat", "sweat", "clammy", "diaphor"]
        ) or ("식은땀" in t)
        has_dyspnea = any(
            x in t for x in ["short of breath", "shortness of breath", "winded", "dyspnea", "숨이", "호흡곤란"]
        )
        has_nausea = any(
            x in t for x in ["nausea", "vomit", "구역", "토"]
        )
        has_syncope = ("syncope" in t) or ("실신" in t)

        risk_factors: List[str] = []
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

        supporting_features: List[str] = []
        if has_chest:
            supporting_features.append("chest discomfort")
        if has_epigastric:
            supporting_features.append("epigastric discomfort")
        if has_diaphoresis:
            supporting_features.append("diaphoresis")
        if has_dyspnea:
            supporting_features.append("dyspnea")
        if has_radiation:
            supporting_features.append("radiation pattern")
        if has_nausea:
            supporting_features.append("nausea")
        if has_syncope:
            supporting_features.append("syncope")

        level = 5
        matched = False

        if (has_chest or has_epigastric) and (has_diaphoresis or has_dyspnea or has_radiation or has_nausea) and high_risk_context:
            matched = True
            level = 2

        if has_chest and (has_diaphoresis or has_syncope) and high_risk_context:
            matched = True
            level = 1

        return {
            "matched": matched,
            "level": level,
            "risk_factors": risk_factors,
            "features": supporting_features,
        }

    def _detect_sirs(self, vitals: Dict[str, Any]) -> Dict[str, Any]:
        hr = _safe_float(vitals.get("hr"))
        rr = _safe_float(vitals.get("rr"))
        temp = _safe_float(vitals.get("temp"))
        spo2 = _safe_float(vitals.get("spo2"))
        sbp, _ = _parse_bp(vitals.get("bp"))

        criteria_hits: List[str] = []

        if temp is not None and (temp > self.th.SIRS_temp_high_f or temp < self.th.SIRS_temp_low_f):
            criteria_hits.append(f"temperature criterion met ({temp:.1f}°F)")
        if hr is not None and hr > self.th.SIRS_hr:
            criteria_hits.append(f"heart rate criterion met ({hr:.0f} bpm)")
        if rr is not None and rr > self.th.SIRS_rr:
            criteria_hits.append(f"respiratory rate criterion met ({rr:.0f}/min)")

        return {
            "sirs_count": len(criteria_hits),
            "criteria_hits": criteria_hits,
            "hr": hr,
            "rr": rr,
            "temp": temp,
            "spo2": spo2,
            "sbp": sbp,
        }

    def _collect_sepsis_signals(self, hub_data: Dict[str, Any]) -> Dict[str, Any]:
        visit = (hub_data or {}).get("visit") or {}
        vitals = (visit.get("vitals") or {}).get("latest") or {}
        transcript = visit.get("transcript") or []
        patient_text = _join_patient_utterances(transcript)

        infection_hits = self._has_any_keyword(patient_text, self.infection_keywords)
        mental_status_hits = self._has_any_keyword(patient_text, self.mental_status_keywords)

        sirs = self._detect_sirs(vitals)
        sirs_count = sirs["sirs_count"]

        has_possible_infection = len(infection_hits) > 0
        has_possible_ams = len(mental_status_hits) > 0
        sbp = sirs["sbp"]
        rr = sirs["rr"]
        spo2 = sirs["spo2"]

        severe_modifiers: List[str] = []
        if sbp is not None and sbp < self.th.Sepsis_SBP_concern:
            severe_modifiers.append(f"hypotension (SBP {sbp})")
        if spo2 is not None and spo2 <= self.th.Sepsis_SpO2_concern:
            severe_modifiers.append(f"oxygenation concern (SpO2 {spo2:.0f}%)")
        if has_possible_ams:
            severe_modifiers.append("possible altered mental status")
        if rr is not None and rr >= self.th.qSOFA_rr:
            severe_modifiers.append(f"qSOFA respiratory criterion (RR {rr:.0f})")
        if sbp is not None and sbp <= self.th.qSOFA_sbp:
            severe_modifiers.append(f"qSOFA blood pressure criterion (SBP {sbp})")

        labs = self._extract_current_labs(hub_data)
        wbc = self._find_lab_value(labs, ["wbc", "white blood cell count", "white blood cells"])
        lactate = self._find_lab_value(labs, ["lactate", "lactic acid"])
        creatinine = self._find_lab_value(labs, ["creatinine", "cr"])

        lab_support: List[str] = []
        if wbc is not None and (wbc >= 12 or wbc <= 4):
            lab_support.append(f"WBC abnormal ({wbc:g})")
        if lactate is not None and lactate >= 2.0:
            lab_support.append(f"lactate elevated ({lactate:g})")
        if creatinine is not None and creatinine >= 1.5:
            lab_support.append(f"creatinine elevated ({creatinine:g})")

        likely_source = "infection"
        t = _norm_text(patient_text)
        if any(x in t for x in ["rlq", "right lower quadrant", "appendic", "우하복부", "충수염"]):
            likely_source = "intra-abdominal source"
        elif any(x in t for x in ["dysuria", "uti", "urinary tract", "배뇨통", "요로감염"]):
            likely_source = "urinary source"
        elif any(x in t for x in ["cough", "phlegm", "pneumonia", "기침", "가래", "폐렴"]):
            likely_source = "pulmonary source"
        elif any(x in t for x in ["cellulitis", "wound", "abscess", "봉와직염", "농양"]):
            likely_source = "skin/soft tissue source"

        matched = False
        level = 5

        if has_possible_infection and sirs_count >= 2:
            matched = True
            level = 2
            if severe_modifiers or lab_support:
                level = 1

        elif has_possible_infection and severe_modifiers:
            matched = True
            level = 2

        return {
            "matched": matched,
            "level": level,
            "infection_hits": infection_hits,
            "mental_status_hits": mental_status_hits,
            "sirs_count": sirs_count,
            "sirs_hits": sirs["criteria_hits"],
            "severe_modifiers": severe_modifiers,
            "lab_support": lab_support,
            "likely_source": likely_source,
        }

    def _collect_general_high_risk_signals(self, hub_data: Dict[str, Any]) -> Dict[str, Any]:
        visit = (hub_data or {}).get("visit") or {}
        transcript = visit.get("transcript") or []
        patient_text = _join_patient_utterances(transcript)
        kw_hits = self._has_any_keyword(patient_text, self.urgent_keywords)

        matched = len(kw_hits) > 0
        level = 2 if matched else 5

        return {
            "matched": matched,
            "level": level,
            "keyword_hits": kw_hits,
        }

    # -----------------------------
    # Optional LLM
    # -----------------------------

    def build_llm_triage_prompt(self, patient_history: str, current_symptoms: str, vitals: Dict[str, Any]) -> str:
        return f"""
You are an expert ER triage assistant.

Patient PMHx: {patient_history}
Current Symptoms: {current_symptoms}
Latest Vitals: {vitals}

Task:
1) Estimate CTAS level (1 to 5).
2) Identify immediate safety concerns.
3) Return concise reasoning.

Respond strictly in this format:
CTAS Level: [1-5]
Concerns:
- ...
Reasons:
- ...
""".strip()

    def evaluate_symptoms_llm(
        self,
        patient_history: str,
        current_symptoms: str,
        vitals: Dict[str, Any],
    ) -> Tuple[int, str, List[str]]:
        if not self.client:
            return 5, "OPENAI_API_KEY missing (LLM triage skipped).", []

        if not (current_symptoms or "").strip():
            return 5, "No recent symptoms reported.", []

        prompt = self.build_llm_triage_prompt(patient_history, current_symptoms, vitals)

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a medical assistant trained in CTAS triage."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=300,
                temperature=0.0,
            )
            txt = (resp.choices[0].message.content or "").strip()

            ctas_level = 5
            reasons = "Unknown"
            concerns: List[str] = []

            m = re.search(r"CTAS\s*Level\s*:\s*(\d)", txt)
            if m:
                ctas_level = int(m.group(1))

            if "Reasons:" in txt:
                reasons = txt.split("Reasons:", 1)[1].strip()

            if "Concerns:" in txt:
                concerns_block = txt.split("Concerns:", 1)[1].split("Reasons:", 1)[0]
                concerns = [line.strip("- ").strip() for line in concerns_block.splitlines() if line.strip()]

            return ctas_level, reasons, concerns
        except Exception as e:  # pragma: no cover
            return 5, f"LLM Evaluation Error: {e}", []

    # -----------------------------
    # Alert synthesis
    # -----------------------------

    def _synthesize_primary_alert(self, hub_data: Dict[str, Any], *, reason: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        visit = hub_data.get("visit") or {}
        vitals = (visit.get("vitals") or {}).get("latest") or {}
        transcript = visit.get("transcript") or []
        patient_text = _join_patient_utterances(transcript)

        vital_info = self._collect_vital_signals(vitals)
        acs_info = self._collect_acs_signals(hub_data)
        sepsis_info = self._collect_sepsis_signals(hub_data)
        general_info = self._collect_general_high_risk_signals(hub_data)

        meta = {
            "reason": reason,
            "ts": time.time(),
            "llm_used": False,
            "vitals_level": vital_info["level"],
            "acs_level": acs_info["level"],
            "sepsis_level": sepsis_info["level"],
            "general_level": general_info["level"],
        }

        alerts: List[Dict[str, Any]] = []

        # Priority 1: Sepsis / severe infection concern
        if sepsis_info["matched"]:
            criteria_parts: List[str] = []

            if sepsis_info["likely_source"] == "intra-abdominal source":
                title = "Possible sepsis from intra-abdominal source"
            elif sepsis_info["likely_source"] == "urinary source":
                title = "Possible sepsis from urinary source"
            elif sepsis_info["likely_source"] == "pulmonary source":
                title = "Possible sepsis from pulmonary source"
            elif sepsis_info["likely_source"] == "skin/soft tissue source":
                title = "Possible sepsis from skin/soft tissue source"
            else:
                title = "Possible sepsis"

            if sepsis_info["infection_hits"]:
                criteria_parts.append(f"Infectious context: {_short_join(sepsis_info['infection_hits'], 4)}")
            if sepsis_info["sirs_count"] >= 2:
                criteria_parts.append(f"SIRS criteria met: {_short_join(sepsis_info['sirs_hits'], 3)}")
            if sepsis_info["severe_modifiers"]:
                criteria_parts.append(f"Physiologic instability: {_short_join(sepsis_info['severe_modifiers'], 4)}")
            if sepsis_info["lab_support"]:
                criteria_parts.append(f"Lab support: {_short_join(sepsis_info['lab_support'], 3)}")

            severity = "high" if sepsis_info["level"] <= 1 else "medium"
            urgency = "immediate" if severity == "high" else "urgent"

            alerts.append(
                self._build_alert(
                    alert_id="sepsis_high_risk",
                    severity=severity,
                    title=title,
                    criteria=". ".join(criteria_parts) if criteria_parts else "High-risk infection pattern detected.",
                    urgency=urgency,
                    source="triage_summary",
                    extras={
                        "pattern": "possible_sepsis",
                        "components": {
                            "vitals": vital_info,
                            "sepsis": sepsis_info,
                        },
                    },
                )
            )
            return alerts, meta

        # Priority 2: ACS / cardiac concern
        if acs_info["matched"]:
            criteria_parts = []
            if acs_info["features"]:
                criteria_parts.append(f"Symptoms: {_short_join(acs_info['features'], 5)}")
            if acs_info["risk_factors"]:
                criteria_parts.append(f"Risk context: {_short_join(acs_info['risk_factors'], 4)}")
            if vital_info["signals"] or vital_info["severe_signals"]:
                all_vitals = vital_info["severe_signals"] + vital_info["signals"]
                criteria_parts.append(f"Associated vital abnormalities: {_short_join(all_vitals, 3)}")

            severity = "high" if acs_info["level"] <= 1 else "medium"
            urgency = "immediate" if severity == "high" else "urgent"

            alerts.append(
                self._build_alert(
                    alert_id="acs_high_risk",
                    severity=severity,
                    title="Possible ACS / time-sensitive cardiac concern",
                    criteria=". ".join(criteria_parts) if criteria_parts else "High-risk ACS symptom pattern detected.",
                    urgency=urgency,
                    source="triage_summary",
                    extras={
                        "pattern": "acs_possible",
                        "components": {
                            "vitals": vital_info,
                            "acs": acs_info,
                        },
                    },
                )
            )
            return alerts, meta

        # Priority 3: severe instability without a named syndrome
        if vital_info["level"] <= 2:
            all_vital_findings = vital_info["severe_signals"] + vital_info["signals"]

            severity = "high" if vital_info["level"] <= 1 else "medium"
            urgency = "immediate" if severity == "high" else "urgent"

            alerts.append(
                self._build_alert(
                    alert_id="physiology_instability",
                    severity=severity,
                    title="Unstable abnormal vital signs",
                    criteria=_short_join(all_vital_findings, 5) or "Abnormal vital signs detected.",
                    urgency=urgency,
                    source="triage_summary",
                    extras={
                        "pattern": "vital_instability",
                        "components": {
                            "vitals": vital_info,
                        },
                    },
                )
            )
            return alerts, meta

        # Priority 4: general high-risk symptom keyword pattern
        if general_info["matched"]:
            alerts.append(
                self._build_alert(
                    alert_id="high_risk_symptom_pattern",
                    severity="medium",
                    title="High-risk symptom pattern",
                    criteria=f"Keywords detected: {_short_join(general_info['keyword_hits'], 5)}",
                    urgency="urgent",
                    source="triage_summary",
                    extras={
                        "pattern": "general_high_risk",
                        "components": {
                            "keywords": general_info,
                        },
                    },
                )
            )
            return alerts, meta

        # Optional LLM fallback only when no rule-based alert fired
        if patient_text.strip() and self.client:
            problems = (hub_data.get("patient") or {}).get("problems") or []
            pmhx = ", ".join([str(x) for x in problems]) if problems else "None"

            ctas_level, ctas_reasons, ctas_concerns = self.evaluate_symptoms_llm(
                pmhx,
                patient_text,
                vitals,
            )
            meta["llm_used"] = True
            meta["ctas"] = ctas_level

            if ctas_level <= 2:
                alerts.append(
                    self._build_alert(
                        alert_id="llm_triage_high_risk",
                        severity="high" if ctas_level == 1 else "medium",
                        title="LLM-flagged high-risk presentation",
                        criteria=ctas_reasons,
                        urgency="immediate" if ctas_level == 1 else "urgent",
                        source="llm_triage",
                        extras={
                            "llm_concerns": ctas_concerns,
                            "pattern": "llm_high_risk",
                        },
                    )
                )
                return alerts, meta

        return [], meta

    # -----------------------------
    # Public API
    # -----------------------------

    def analyze(self, hub_data: Dict[str, Any], *, reason: str = "periodic") -> Dict[str, Any]:
        hub_data = hub_data or {}

        alerts, meta = self._synthesize_primary_alert(hub_data, reason=reason)

        meta["alert_count"] = len(alerts)

        return {
            "alerts": alerts,
            "status": "done",
            "meta": meta,
        }