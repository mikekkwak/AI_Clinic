import time


class AIDoctor:
    """
    Doctor 모듈:
    - 핵심 syndrome alert 1개
    - recommendations
    - recommended orders (향후 EMR 연동 버튼/체크박스와 연결하기 쉽게)
    - clinical note (CC는 요약)
    """

    def _collect_patient_utterances(self, transcript):
        pts = []
        for msg in transcript or []:
            if isinstance(msg, dict) and msg.get("role") == "patient":
                t = (msg.get("text") or "").strip()
                if t:
                    pts.append(t)
        return " ".join(pts)

    def _summarize_cc(self, patient_text: str) -> str:
        t = (patient_text or "").lower()

        parts = []

        # location / quality
        if any(k in t for k in ["upper", "epigastr", "below my ribs", "stomach", "상복부", "명치", "위"]):
            parts.append("epigastric/upper abdominal discomfort")
        if any(k in t for k in ["pressure", "pressing", "heavy", "dull", "압박", "무거", "눌리는"]):
            parts.append("pressure/heaviness quality")

        # associated
        if any(k in t for k in ["nausea", "vomit", "메스껍", "구역"]):
            parts.append("nausea")
        if any(k in t for k in ["cold sweat", "clammy", "diaphores", "식은땀", "땀"]):
            parts.append("diaphoresis/clamminess")
        if any(k in t for k in ["winded", "short of breath", "dyspnea", "숨", "호흡곤란"]):
            parts.append("exertional dyspnea")

        # radiation
        if any(k in t for k in ["shoulder", "arm", "radiat", "어깨", "팔"]):
            parts.append("possible radiation")

        if not parts:
            return "symptoms reported (see HPI)"
        return "; ".join(parts)

    def analyze(self, hub_snapshot: dict, reason: str = "periodic") -> dict:
        patient = hub_snapshot.get("patient", {}) or {}
        visit = hub_snapshot.get("visit", {}) or {}
        transcript = visit.get("transcript", []) or []
        vitals = (visit.get("vitals", {}) or {}).get("latest") or (visit.get("vitals", {}) or {})

        profile = patient.get("profile", {}) or {}
        problems = patient.get("problems", []) or []
        allergies = patient.get("allergies", []) or []

        patient_text = self._collect_patient_utterances(transcript)
        cc_summary = self._summarize_cc(patient_text)

        # 간단 risk context
        age = profile.get("age")
        has_dm = any("dm" in (p.lower()) or "diabetes" in (p.lower()) for p in problems)
        high_risk_context = bool((age and int(age) >= 60) or has_dm)

        # 핵심: 딱 1개 alert
        alerts = []
        if high_risk_context and any(k in patient_text.lower() for k in ["pressure", "heavy", "cold sweat", "clammy", "winded", "nausea", "shoulder"]):
            alerts.append(
                {
                    "severity": "high",
                    "title": "Possible Atypical ACS / MI",
                    "criteria": "High-risk context (age/DM) + epigastric pressure/heaviness with diaphoresis/nausea/dyspnea ± radiation.",
                }
            )

        recommendations = []
        if alerts:
            recommendations = [
                "Treat as possible ACS until ruled out; atypical presentations are common in elderly/diabetic/women.",
                "Obtain ECG early; repeat ECGs if symptoms persist or initial ECG is nondiagnostic.",
                "Send serial troponins; if ischemic ECG or positive biomarkers: activate ACS pathway / cardiology per local protocol.",
            ]

        # ✅ orders는 너무 많은 카테고리 대신, 1개 세트로 간단히
        suggested_orders = []
        if alerts:
            suggested_orders = [
                {
                    "title": "ACS Rule-out (STAT)",
                    "items": [
                        "12-lead ECG now (repeat if needed)",
                        "Cardiac monitor + IV access",
                        "Troponin series (0 and repeat per protocol), CMP, CBC",
                        "Consider CXR if dyspnea/alternative dx concern",
                        "Aspirin if no contraindication",
                    ],
                }
            ]

        # Clinical Note (CC에 환자 말 그대로가 아니라 요약)
        bp = vitals.get("bp", "-")
        hr = vitals.get("hr", "-")
        rr = vitals.get("rr", "-")
        spo2 = vitals.get("spo2", "-")
        temp = vitals.get("temp", "-")

        pmhx_str = ", ".join(problems) if problems else "None"
        alg_str = ", ".join(allergies) if allergies else "None"

        note = f"""Subjective
- CC: {cc_summary}
- PMHx: {pmhx_str}
- Allergies: {alg_str}

Objective
- Vitals: BP {bp}, HR {hr}, RR {rr}, SpO2 {spo2}, Temp {temp}

Assessment
- {alerts[0]["title"] if alerts else "No high-risk syndrome detected yet."}

Plan
- See Recommendations / Orders.
"""

        return {
            "note": note,
            "alerts": alerts,
            "recommendations": recommendations,
            "suggested_orders": suggested_orders,
            "meta": {"reason": reason, "timestamp": time.time()},
        }