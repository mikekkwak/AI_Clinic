import time
from typing import Any, Dict, List, Optional


class AIDoctor:
    """
    Doctor module

    역할
    - sync 전에도 recommendation 생성
    - triage alert review
    - assessment / ddx 구조화
    - recommendations
    - recommended_orders (향후 EMR 연동용)
    - suggested_orders (현재 UI 호환용)
    - note_inputs + backward-compatible note text
    """

    # -----------------------------
    # Basic helpers
    # -----------------------------

    def _safe_lower(self, x: Any) -> str:
        if x is None:
            return ""
        return str(x).lower()

    def _collect_patient_utterances(self, transcript: List[Dict[str, Any]]) -> str:
        pts = []
        for msg in transcript or []:
            if isinstance(msg, dict) and msg.get("role") == "patient":
                t = (msg.get("text") or "").strip()
                if t:
                    pts.append(t)
        return " ".join(pts)

    def _get_profile(self, hub_snapshot: dict) -> Dict[str, Any]:
        return (hub_snapshot.get("patient") or {}).get("profile") or {}

    def _get_problems(self, hub_snapshot: dict) -> List[str]:
        patient = hub_snapshot.get("patient", {}) or {}
        return patient.get("problems") or patient.get("pmhx") or []

    def _get_allergies(self, hub_snapshot: dict) -> List[str]:
        return (hub_snapshot.get("patient") or {}).get("allergies") or []

    def _get_medications(self, hub_snapshot: dict) -> List[str]:
        return (hub_snapshot.get("patient") or {}).get("medications") or []

    def _get_vitals(self, hub_snapshot: dict) -> Dict[str, Any]:
        visit = hub_snapshot.get("visit", {}) or {}
        raw = (visit.get("vitals", {}) or {}).get("latest") or (visit.get("vitals", {}) or {})

        return {
            "bp": raw.get("bp", ""),
            "hr": raw.get("hr", ""),
            "rr": raw.get("rr", ""),
            "spo2": raw.get("spo2") or raw.get("spo2(%)") or "",
            "temp": raw.get("temp") or raw.get("temp(F)") or "",
        }

    def _get_latest_labs(self, hub_snapshot: dict) -> List[Dict[str, Any]]:
        visit = hub_snapshot.get("visit", {}) or {}
        labs = (visit.get("labs") or {}).get("latest") or []
        return labs if isinstance(labs, list) else []

    def _get_triage_alerts(self, hub_snapshot: dict) -> List[Dict[str, Any]]:
        intelligence = hub_snapshot.get("intelligence", {}) or {}
        triage = (intelligence.get("triage") or {}).get("alerts") or []
        return triage if isinstance(triage, list) else []

    def _get_care_setting(self, hub_snapshot: dict) -> str:
        """
        기본은 urgent care.
        나중에 app.py / note template에서 meta.care_setting = 'ed' 로 넘기면 ED 분기 가능.
        """
        meta = hub_snapshot.get("meta", {}) or {}
        care_setting = self._safe_lower(meta.get("care_setting") or "uc").strip()
        return "ed" if care_setting == "ed" else "uc"

    def _parse_float(self, x: Any) -> Optional[float]:
        try:
            if x is None or str(x).strip() == "":
                return None
            return float(str(x).replace("%", "").replace("F", "").replace("f", "").strip())
        except Exception:
            return None

    def _parse_temp_f(self, temp_raw: Any) -> Optional[float]:
        if temp_raw is None:
            return None
        s = str(temp_raw).strip().lower().replace(" ", "")
        try:
            # "39.2C" 같은 경우
            if s.endswith("c"):
                c = float(s[:-1])
                return c * 9 / 5 + 32
            # "102.6F" 같은 경우
            if s.endswith("f"):
                return float(s[:-1])
            # 숫자만 있는 경우 F로 간주
            return float(s)
        except Exception:
            return None

    def _lab_match(self, lab_name: str, targets: List[str]) -> bool:
        nm = self._safe_lower(lab_name)
        return any(t in nm for t in targets)

    def _find_lab(self, labs: List[Dict[str, Any]], targets: List[str]) -> Optional[Dict[str, Any]]:
        for lab in labs or []:
            name = lab.get("name") or ""
            if self._lab_match(name, [t.lower() for t in targets]):
                return lab
        return None

    def _summarize_cc(self, patient_text: str) -> str:
        t = self._safe_lower(patient_text)

        # ACS-like
        if any(k in t for k in ["epigastr", "upper stomach", "below my ribs", "stomach", "명치", "상복부"]):
            if any(k in t for k in ["pressure", "pressing", "heavy", "dull", "압박", "눌리는", "무거"]):
                return "epigastric pressure-like pain"
            return "epigastric pain"

        # RLQ / appendicitis-like
        if any(k in t for k in ["lower right", "right lower", "rlq", "오른쪽 아랫", "우하복부"]):
            return "right lower quadrant abdominal pain"

        # UTI-like
        if any(k in t for k in ["burning when i urinate", "burning when i pee", "dysuria", "urinate", "pee", "배뇨통"]):
            return "dysuria / urinary discomfort"

        if not t.strip():
            return "symptoms reported"
        return "symptoms reported (see HPI)"

    # -----------------------------
    # Clinical signal detection
    # -----------------------------

    def _acs_concern(self, patient_text: str, problems: List[str], labs: List[Dict[str, Any]]) -> bool:
        t = self._safe_lower(patient_text)
        ptxt = " ".join([self._safe_lower(p) for p in problems])

        feature_count = 0
        if any(k in t for k in ["epigastr", "upper stomach", "below my ribs", "stomach", "chest pressure", "chest pain", "pressure", "heavy"]):
            feature_count += 1
        if any(k in t for k in ["clammy", "cold sweat", "diaphores", "식은땀"]):
            feature_count += 1
        if any(k in t for k in ["nausea", "nauseous", "구역", "메스껍"]):
            feature_count += 1
        if any(k in t for k in ["short of breath", "winded", "dyspnea", "숨"]):
            feature_count += 1
        if any(k in t for k in ["shoulder", "arm", "radiat", "어깨", "팔"]):
            feature_count += 1
        if any(k in ptxt for k in ["dm", "diabetes", "type 2 dm"]):
            feature_count += 1

        troponin = self._find_lab(labs, ["troponin"])
        troponin_high = False
        if troponin:
            flag = self._safe_lower(troponin.get("flag"))
            if flag in ("high", "critical"):
                troponin_high = True
            else:
                val = self._parse_float(troponin.get("value"))
                if val is not None and val > 0.04:
                    troponin_high = True

        if troponin_high:
            feature_count += 2

        return feature_count >= 3

    def _appendicitis_concern(self, patient_text: str) -> bool:
        t = self._safe_lower(patient_text)

        score = 0
        if any(k in t for k in ["belly button", "around my belly button", "배꼽"]):
            score += 1
        if any(k in t for k in ["lower right", "right lower", "rlq", "오른쪽 아랫", "우하복부"]):
            score += 1
        if any(k in t for k in ["worse when i walk", "walking makes it worse", "move", "movement", "움직이면", "걸으면"]):
            score += 1
        if any(k in t for k in ["vomit", "threw up", "nausea", "구토", "메스껍"]):
            score += 1

        return score >= 3

    def _sepsis_concern(self, patient_text: str, vitals: Dict[str, Any], labs: List[Dict[str, Any]]) -> bool:
        t = self._safe_lower(patient_text)

        fever_symptom = any(k in t for k in ["fever", "chills", "shaking chills", "열", "오한"])
        hr = self._parse_float(vitals.get("hr"))
        bp_raw = str(vitals.get("bp") or "")
        temp_f = self._parse_temp_f(vitals.get("temp"))

        hypotension = False
        if "/" in bp_raw:
            try:
                sys_bp = float(bp_raw.split("/")[0].strip())
                if sys_bp < 100:
                    hypotension = True
            except Exception:
                pass

        tachy = hr is not None and hr >= 110
        fever_vital = temp_f is not None and temp_f >= 100.4

        wbc = self._find_lab(labs, ["wbc"])
        lactate = self._find_lab(labs, ["lactate"])

        wbc_high = False
        if wbc:
            flag = self._safe_lower(wbc.get("flag"))
            if flag in ("high", "critical"):
                wbc_high = True
            else:
                val = self._parse_float(wbc.get("value"))
                if val is not None and val >= 12:
                    wbc_high = True

        lactate_high = False
        if lactate:
            flag = self._safe_lower(lactate.get("flag"))
            if flag in ("high", "critical"):
                lactate_high = True
            else:
                val = self._parse_float(lactate.get("value"))
                if val is not None and val >= 2:
                    lactate_high = True

        score = 0
        if fever_symptom or fever_vital:
            score += 1
        if tachy:
            score += 1
        if hypotension:
            score += 1
        if wbc_high:
            score += 1
        if lactate_high:
            score += 2

        return score >= 3

    def _uti_concern(self, patient_text: str, labs: List[Dict[str, Any]]) -> bool:
        t = self._safe_lower(patient_text)
        urinary_symptoms = any(k in t for k in ["burning when i urinate", "burning when i pee", "dysuria", "frequency", "urgency", "urinate", "pee"])
        nitrite = self._find_lab(labs, ["urinalysis - nitrite", "nitrite"])
        leuk = self._find_lab(labs, ["urinalysis - leukocyte esterase", "leukocyte esterase"])
        ua_wbc = self._find_lab(labs, ["urinalysis - wbc", "ua wbc"])

        ua_support = any([nitrite, leuk, ua_wbc])
        return urinary_symptoms or ua_support

    def _pyelo_possible(self, patient_text: str) -> bool:
        t = self._safe_lower(patient_text)
        return any(k in t for k in ["flank pain", "back pain", "side pain", "옆구리"])

    # -----------------------------
    # Triage alert review
    # -----------------------------

    def _review_triage_alerts(
        self,
        triage_alerts: List[Dict[str, Any]],
        acs: bool,
        sepsis: bool,
        appendicitis: bool,
        uti: bool,
    ) -> List[Dict[str, Any]]:
        reviews: List[Dict[str, Any]] = []

        for alert in triage_alerts or []:
            title = self._safe_lower(alert.get("title"))
            alert_id = alert.get("id") or title.replace(" ", "_")

            status = "supported"
            comment = ""

            if "acs" in title or "mi" in title or "cardiac" in title:
                if acs:
                    status = "supported"
                    comment = "Features remain concerning for ACS."
                else:
                    status = "downgraded"
                    comment = "Current data do not strongly support ACS."
            elif "sepsis" in title:
                if sepsis:
                    status = "supported"
                    comment = "Sepsis concern remains significant."
                else:
                    status = "dismissed"
                    comment = "Current data do not strongly support sepsis."
            elif "abdomen" in title or "surgical" in title or "appendic" in title:
                if appendicitis:
                    status = "supported"
                    comment = "Current findings support intra-abdominal pathology concern."
                else:
                    status = "downgraded"
                    comment = "Current findings are less supportive."
            elif "uti" in title or "pyelo" in title:
                if uti:
                    status = "supported"
                    comment = "Urinary source remains plausible."
                else:
                    status = "dismissed"
                    comment = "Urinary source not well supported."

            reviews.append(
                {
                    "alert_id": alert_id,
                    "status": status,
                    "comment": comment,
                }
            )

        return reviews

    # -----------------------------
    # Assessment / recommendations / orders
    # -----------------------------

    def _build_assessment(
        self,
        acs: bool,
        sepsis: bool,
        appendicitis: bool,
        uti: bool,
        pyelo_possible: bool,
    ) -> List[Dict[str, str]]:
        assessment: List[Dict[str, str]] = []

        if acs:
            assessment.append(
                {
                    "name": "Acute coronary syndrome",
                    "likelihood": "moderate",
                    "risk": "high",
                    "kind": "working_diagnosis",
                }
            )
            assessment.append(
                {
                    "name": "Gastritis / dyspepsia",
                    "likelihood": "low",
                    "risk": "low",
                    "kind": "ddx",
                }
            )
            return assessment

        if sepsis:
            assessment.append(
                {
                    "name": "Sepsis due to intra-abdominal infection",
                    "likelihood": "high",
                    "risk": "high",
                    "kind": "working_diagnosis",
                }
            )
            if appendicitis:
                assessment.append(
                    {
                        "name": "Appendicitis with perforation concern",
                        "likelihood": "moderate",
                        "risk": "high",
                        "kind": "working_diagnosis",
                    }
                )
            return assessment

        if uti:
            assessment.append(
                {
                    "name": "Uncomplicated cystitis",
                    "likelihood": "high" if not pyelo_possible else "moderate",
                    "risk": "low",
                    "kind": "working_diagnosis",
                }
            )
            assessment.append(
                {
                    "name": "Pyelonephritis",
                    "likelihood": "low" if not pyelo_possible else "moderate",
                    "risk": "moderate",
                    "kind": "ddx",
                }
            )
            return assessment

        assessment.append(
            {
                "name": "Undifferentiated symptoms",
                "likelihood": "moderate",
                "risk": "moderate",
                "kind": "working_diagnosis",
            }
        )
        return assessment

    def _build_recommendations(
        self,
        care_setting: str,
        acs: bool,
        sepsis: bool,
        appendicitis: bool,
        uti: bool,
        pyelo_possible: bool,
    ) -> List[str]:
        recs: List[str] = []

        if acs:
            recs.append("Immediate ECG")
            recs.append("Repeat troponin")
            if care_setting == "uc":
                recs.append("Emergency department evaluation recommended")
            else:
                recs.append("Continue cardiac evaluation")
            return recs

        if sepsis:
            recs.append("Fluid resuscitation")
            recs.append("Broad-spectrum antibiotics should be considered")
            if appendicitis:
                recs.append("Urgent surgical evaluation")
            if care_setting == "uc":
                recs.append("Emergency department evaluation recommended")
            else:
                recs.append("Escalation of care recommended")
            return recs

        if uti:
            recs.append("Urinalysis findings support urinary tract infection")
            if pyelo_possible:
                recs.append("Escalate if fever, flank pain, vomiting, or worsening symptoms")
            else:
                recs.append("Nitrofurantoin is reasonable if uncomplicated cystitis is suspected")
                recs.append("Return if fever, flank pain, vomiting, or worsening symptoms")
            return recs

        recs.append("Reassess as additional information becomes available")
        return recs

    def _build_recommended_orders(
        self,
        care_setting: str,
        acs: bool,
        sepsis: bool,
        appendicitis: bool,
        uti: bool,
        pyelo_possible: bool,
    ) -> List[Dict[str, str]]:
        orders: List[Dict[str, str]] = []

        if acs:
            orders.extend(
                [
                    {
                        "id": "ecg_12_lead",
                        "label": "ECG 12-lead",
                        "type": "procedure",
                        "priority": "high",
                        "reason": "Possible ACS",
                    },
                    {
                        "id": "troponin_i",
                        "label": "Troponin-I",
                        "type": "lab",
                        "priority": "high",
                        "reason": "Possible ACS",
                    },
                    {
                        "id": "cbc",
                        "label": "CBC",
                        "type": "lab",
                        "priority": "medium",
                        "reason": "Cardiac workup",
                    },
                    {
                        "id": "cmp",
                        "label": "CMP",
                        "type": "lab",
                        "priority": "medium",
                        "reason": "Cardiac workup",
                    },
                ]
            )
            return orders

        if sepsis:
            orders.extend(
                [
                    {
                        "id": "cbc",
                        "label": "CBC",
                        "type": "lab",
                        "priority": "high",
                        "reason": "Sepsis concern",
                    },
                    {
                        "id": "cmp",
                        "label": "CMP",
                        "type": "lab",
                        "priority": "high",
                        "reason": "Sepsis concern",
                    },
                    {
                        "id": "lactate",
                        "label": "Lactate",
                        "type": "lab",
                        "priority": "high",
                        "reason": "Sepsis concern",
                    },
                    {
                        "id": "blood_cultures",
                        "label": "Blood cultures",
                        "type": "lab",
                        "priority": "high",
                        "reason": "Possible bacteremia",
                    },
                    {
                        "id": "iv_fluids",
                        "label": "IV fluids",
                        "type": "procedure",
                        "priority": "high",
                        "reason": "Hypoperfusion / sepsis concern",
                    },
                    {
                        "id": "broad_spectrum_antibiotics",
                        "label": "Broad-spectrum antibiotics",
                        "type": "medication",
                        "priority": "high",
                        "reason": "Sepsis concern",
                    },
                ]
            )
            if appendicitis:
                orders.append(
                    {
                        "id": "ct_abdomen_pelvis",
                        "label": "CT abdomen/pelvis",
                        "type": "imaging",
                        "priority": "high",
                        "reason": "Possible appendicitis / perforation",
                    }
                )
            return orders

        if uti:
            orders.extend(
                [
                    {
                        "id": "urinalysis",
                        "label": "Urinalysis",
                        "type": "lab",
                        "priority": "medium",
                        "reason": "Urinary symptoms",
                    },
                    {
                        "id": "urine_culture",
                        "label": "Urine culture",
                        "type": "lab",
                        "priority": "medium",
                        "reason": "UTI evaluation",
                    },
                ]
            )
            if pyelo_possible:
                orders.append(
                    {
                        "id": "cbc",
                        "label": "CBC",
                        "type": "lab",
                        "priority": "medium",
                        "reason": "Evaluate for upper urinary tract infection",
                    }
                )
            return orders

        return orders

    def _orders_for_current_ui(self, recommended_orders: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        현재 app.py는 suggested_orders = [{"title":..., "items":[...]}] 형태를 기대함.
        내부 표준 recommended_orders를 현재 UI 호환 형식으로 변환.
        """
        if not recommended_orders:
            return []

        high_items = [o["label"] for o in recommended_orders if o.get("priority") == "high"]
        medium_items = [o["label"] for o in recommended_orders if o.get("priority") != "high"]

        blocks = []
        if high_items:
            blocks.append({"title": "Priority Orders", "priority": "high", "items": high_items})
        if medium_items:
            blocks.append({"title": "Additional Orders", "priority": "medium", "items": medium_items})
        return blocks

    def _doctor_alerts(
        self,
        care_setting: str,
        acs: bool,
        sepsis: bool,
        appendicitis: bool,
    ) -> List[Dict[str, Any]]:
        alerts: List[Dict[str, Any]] = []

        if acs:
            alerts.append(
                {
                    "severity": "high",
                    "title": "ACS concern remains high",
                    "criteria": "Symptoms and available data remain concerning for ACS.",
                }
            )
            return alerts

        if sepsis:
            title = "Possible sepsis"
            criteria = "Current findings suggest high-risk infection."
            if appendicitis:
                title = "Possible sepsis from intra-abdominal source"
                criteria = "Findings raise concern for appendicitis with intra-abdominal infection."
            alerts.append(
                {
                    "severity": "high",
                    "title": title,
                    "criteria": criteria,
                }
            )
            return alerts

        return alerts

    def _disposition(
        self,
        care_setting: str,
        acs: bool,
        sepsis: bool,
        appendicitis: bool,
        uti: bool,
        pyelo_possible: bool,
    ) -> Dict[str, str]:
        if acs:
            if care_setting == "uc":
                return {
                    "level": "ed_transfer",
                    "text": "Emergency department evaluation recommended",
                }
            return {
                "level": "observation",
                "text": "Continue ED cardiac evaluation",
            }

        if sepsis:
            if care_setting == "uc":
                return {
                    "level": "ed_transfer",
                    "text": "Emergency department evaluation recommended",
                }
            return {
                "level": "admit",
                "text": "Admission-level care should be considered",
            }

        if uti:
            if pyelo_possible:
                return {
                    "level": "uncertain",
                    "text": "Escalation may be needed if symptoms worsen",
                }
            return {
                "level": "outpatient",
                "text": "Outpatient treatment appears reasonable",
            }

        return {
            "level": "uncertain",
            "text": "Disposition depends on reassessment",
        }

    def _note_inputs(
        self,
        cc_summary: str,
        assessment: List[Dict[str, str]],
        recommendations: List[str],
        disposition: Dict[str, str],
        care_setting: str,
        uti: bool,
        pyelo_possible: bool,
    ) -> Dict[str, Any]:
        working_dx = assessment[0]["name"] if assessment else "Undifferentiated symptoms"
        return_precautions: List[str] = []

        if uti:
            return_precautions = ["fever", "flank pain", "vomiting", "worsening symptoms"]
        elif care_setting == "uc" and disposition.get("level") == "outpatient":
            return_precautions = ["worsening symptoms"]

        mdm = working_dx
        if disposition.get("text"):
            mdm = f"{working_dx}. {disposition.get('text')}."

        return {
            "summary": cc_summary,
            "mdm": mdm,
            "return_precautions": return_precautions,
        }

    def _legacy_note_text(
        self,
        cc_summary: str,
        problems: List[str],
        allergies: List[str],
        vitals: Dict[str, Any],
        assessment: List[Dict[str, str]],
        recommendations: List[str],
    ) -> str:
        bp = vitals.get("bp", "-") or "-"
        hr = vitals.get("hr", "-") or "-"
        rr = vitals.get("rr", "-") or "-"
        spo2 = vitals.get("spo2", "-") or "-"
        temp = vitals.get("temp", "-") or "-"

        pmhx_str = ", ".join(problems) if problems else "None"
        alg_str = ", ".join(allergies) if allergies else "None"
        assessment_line = assessment[0]["name"] if assessment else "No working diagnosis"
        plan_text = "; ".join(recommendations[:3]) if recommendations else "Continue reassessment."

        note = f"""Subjective
- CC: {cc_summary}
- PMHx: {pmhx_str}
- Allergies: {alg_str}

Objective
- Vitals: BP {bp}, HR {hr}, RR {rr}, SpO2 {spo2}, Temp {temp}

Assessment
- {assessment_line}

Plan
- {plan_text}
"""
        return note

    # -----------------------------
    # Main analysis
    # -----------------------------

    def analyze(self, hub_snapshot: dict, reason: str = "periodic") -> dict:
        patient = hub_snapshot.get("patient", {}) or {}
        visit = hub_snapshot.get("visit", {}) or {}
        transcript = visit.get("transcript", []) or []

        profile = self._get_profile(hub_snapshot)
        problems = self._get_problems(hub_snapshot)
        allergies = self._get_allergies(hub_snapshot)
        medications = self._get_medications(hub_snapshot)
        vitals = self._get_vitals(hub_snapshot)
        labs = self._get_latest_labs(hub_snapshot)
        triage_alerts = self._get_triage_alerts(hub_snapshot)
        care_setting = self._get_care_setting(hub_snapshot)

        patient_text = self._collect_patient_utterances(transcript)
        cc_summary = self._summarize_cc(patient_text)

        # Case detection
        acs = self._acs_concern(patient_text, problems, labs)
        appendicitis = self._appendicitis_concern(patient_text)
        sepsis = self._sepsis_concern(patient_text, vitals, labs)
        uti = self._uti_concern(patient_text, labs)
        pyelo_possible = self._pyelo_possible(patient_text)

        triage_alert_review = self._review_triage_alerts(
            triage_alerts=triage_alerts,
            acs=acs,
            sepsis=sepsis,
            appendicitis=appendicitis,
            uti=uti,
        )

        assessment = self._build_assessment(
            acs=acs,
            sepsis=sepsis,
            appendicitis=appendicitis,
            uti=uti,
            pyelo_possible=pyelo_possible,
        )

        recommendations = self._build_recommendations(
            care_setting=care_setting,
            acs=acs,
            sepsis=sepsis,
            appendicitis=appendicitis,
            uti=uti,
            pyelo_possible=pyelo_possible,
        )

        recommended_orders = self._build_recommended_orders(
            care_setting=care_setting,
            acs=acs,
            sepsis=sepsis,
            appendicitis=appendicitis,
            uti=uti,
            pyelo_possible=pyelo_possible,
        )

        suggested_orders = self._orders_for_current_ui(recommended_orders)

        doctor_alerts = self._doctor_alerts(
            care_setting=care_setting,
            acs=acs,
            sepsis=sepsis,
            appendicitis=appendicitis,
        )

        disposition = self._disposition(
            care_setting=care_setting,
            acs=acs,
            sepsis=sepsis,
            appendicitis=appendicitis,
            uti=uti,
            pyelo_possible=pyelo_possible,
        )

        note_inputs = self._note_inputs(
            cc_summary=cc_summary,
            assessment=assessment,
            recommendations=recommendations,
            disposition=disposition,
            care_setting=care_setting,
            uti=uti,
            pyelo_possible=pyelo_possible,
        )

        note = self._legacy_note_text(
            cc_summary=cc_summary,
            problems=problems,
            allergies=allergies,
            vitals=vitals,
            assessment=assessment,
            recommendations=recommendations,
        )

        return {
            # backward compatibility
            "note": note,
            "alerts": doctor_alerts,
            "recommendations": recommendations,
            "suggested_orders": suggested_orders,

            # new structured outputs
            "triage_alert_review": triage_alert_review,
            "assessment": assessment,
            "recommended_orders": recommended_orders,
            "disposition": disposition,
            "note_inputs": note_inputs,

            "meta": {
                "reason": reason,
                "timestamp": time.time(),
                "mode": "rule_based",
                "care_setting": care_setting,
                "known_data": {
                    "has_transcript": bool(patient_text.strip()),
                    "has_vitals": any(bool(str(v).strip()) for v in vitals.values()),
                    "has_labs": bool(labs),
                    "has_pmhx": bool(problems),
                    "has_medications": bool(medications),
                },
            },
        }