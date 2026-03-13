from typing import Dict, Any, List, Optional


class NoteEngine:
    """
    Deterministic note composer.

    목적:
    - hub snapshot(ctx)을 받아
    - preset template (UC / ED)로
    - 복붙 가능한 note text 생성
    - LLM 없이 안정적으로 동작
    """

    def compose_note(self, ctx: Dict[str, Any], template: str = "uc_dr_kim") -> str:
        if template == "ed_standard":
            return self._compose_ed_note(ctx)
        return self._compose_uc_note(ctx)

    # -----------------------------
    # Basic extractors
    # -----------------------------

    def _patient_profile(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        return (ctx.get("patient") or {}).get("profile") or {}

    def _patient_lines(self, ctx: Dict[str, Any]) -> List[str]:
        transcript = ((ctx.get("visit") or {}).get("transcript") or [])
        out: List[str] = []
        for msg in transcript:
            if (msg.get("role") or "").lower() == "patient":
                txt = (msg.get("text") or "").strip()
                if txt:
                    out.append(txt)
        return out

    def _doctor_lines(self, ctx: Dict[str, Any]) -> List[str]:
        transcript = ((ctx.get("visit") or {}).get("transcript") or [])
        out: List[str] = []
        for msg in transcript:
            if (msg.get("role") or "").lower() == "doctor":
                txt = (msg.get("text") or "").strip()
                if txt:
                    out.append(txt)
        return out

    def _pmhx_list(self, ctx: Dict[str, Any]) -> List[str]:
        return (ctx.get("patient") or {}).get("problems") or []

    def _pmhx(self, ctx: Dict[str, Any]) -> str:
        probs = self._pmhx_list(ctx)
        if not probs:
            return "None reported"
        return ", ".join(probs)

    def _allergies(self, ctx: Dict[str, Any]) -> str:
        a = (ctx.get("patient") or {}).get("allergies") or []
        if not a:
            return "NKDA"
        return ", ".join(a)

    def _meds(self, ctx: Dict[str, Any]) -> str:
        meds = (ctx.get("patient") or {}).get("medications") or []
        if not meds:
            return "None listed"
        return ", ".join(meds)

    def _vitals(self, ctx: Dict[str, Any]) -> str:
        v = ((ctx.get("visit") or {}).get("vitals") or {}).get("latest") or {}
        if not v:
            return "Not available"

        bp = v.get("bp", "-")
        hr = v.get("hr", "-")
        rr = v.get("rr", "-")
        spo2 = v.get("spo2", "-")
        temp = v.get("temp", "-")

        return f"BP {bp}, HR {hr}, RR {rr}, SpO2 {spo2}, Temp {temp}"

    def _alerts(self, ctx: Dict[str, Any]) -> List[Dict[str, Any]]:
        triage = ((ctx.get("intelligence") or {}).get("triage") or {}).get("alerts") or []
        doctor = ((ctx.get("intelligence") or {}).get("doctor") or {}).get("alerts") or []
        return triage + doctor

    def _alert_titles(self, ctx: Dict[str, Any]) -> List[str]:
        out: List[str] = []
        for a in self._alerts(ctx):
            title = (a.get("title") or "").strip()
            if title:
                out.append(title)
        return out

    def _orders(self, ctx: Dict[str, Any]) -> List[str]:
        orders = ((ctx.get("intelligence") or {}).get("doctor") or {}).get("suggested_orders") or []
        result: List[str] = []
        for o in orders:
            items = o.get("items") or []
            for item in items:
                if item and item not in result:
                    result.append(item)
        return result

    def _critical_labs(self, ctx: Dict[str, Any]) -> List[Dict[str, Any]]:
        return ((ctx.get("visit") or {}).get("labs") or {}).get("critical") or []

    def _labs_summary(self, ctx: Dict[str, Any]) -> str:
        crit = self._critical_labs(ctx)
        if not crit:
            return "No critical labs."

        parts = []
        for c in crit:
            name = c.get("name") or "Lab"
            val = c.get("value", "")
            unit = c.get("unit", "")
            flag = c.get("flag", "")
            piece = f"{name} {val}"
            if unit:
                piece += f" {unit}"
            if flag:
                piece += f" ({flag})"
            parts.append(piece)
        return ", ".join(parts)

    # -----------------------------
    # Clinical reasoning helpers
    # -----------------------------

    def _full_patient_text(self, ctx: Dict[str, Any]) -> str:
        return " ".join(self._patient_lines(ctx)).lower()

    def _has_phrase(self, ctx: Dict[str, Any], phrases: List[str]) -> bool:
        txt = self._full_patient_text(ctx)
        return any(p.lower() in txt for p in phrases)

    def _troponin_high(self, ctx: Dict[str, Any]) -> bool:
        for lab in self._critical_labs(ctx):
            name = (lab.get("name") or "").lower()
            flag = (lab.get("flag") or "").lower()

            if "troponin" in name and flag in ("high", "critical"):
                return True

            if "troponin" in name:
                try:
                    val = float(lab.get("value"))
                    if val > 0.04:
                        return True
                except Exception:
                    pass
        return False

    def _acs_concern(self, ctx: Dict[str, Any]) -> bool:
        titles = " ".join(self._alert_titles(ctx)).lower()
        patient_txt = self._full_patient_text(ctx)
        pmhx = " ".join(self._pmhx_list(ctx)).lower()

        if "acs" in titles or "cardiac" in titles:
            return True

        strong_features = 0

        if any(w in patient_txt for w in ["epigastric", "upper stomach", "below my ribs", "stomach"]):
            strong_features += 1
        if any(w in patient_txt for w in ["shoulder", "left shoulder"]):
            strong_features += 1
        if any(w in patient_txt for w in ["clammy", "cold sweat", "sweat", "diaphoresis"]):
            strong_features += 1
        if any(w in patient_txt for w in ["short of breath", "winded", "sob"]):
            strong_features += 1
        if any(w in pmhx for w in ["dm", "diabetes"]):
            strong_features += 1
        if self._troponin_high(ctx):
            strong_features += 2

        return strong_features >= 3

    # -----------------------------
    # Text generation helpers
    # -----------------------------

    def _first_patient_statement(self, ctx: Dict[str, Any]) -> str:
        lines = self._patient_lines(ctx)
        if not lines:
            return "Patient presents for evaluation."
        return lines[0]

    def _chief_complaint(self, ctx: Dict[str, Any]) -> str:
        """
        chief complaint는 짧고 chart-friendly 하게.
        """
        txt = self._full_patient_text(ctx)

        if any(p in txt for p in ["chest pressure", "chest tightness", "chest pain"]):
            return "Chest pain / chest pressure"

        if any(p in txt for p in ["epigastric", "upper stomach", "upper part", "below my ribs", "stomach"]):
            if any(p in txt for p in ["pressure", "heavy", "pressing down"]):
                return "Epigastric pain / pressure"
            return "Epigastric pain"

        if "short of breath" in txt or "winded" in txt:
            return "Shortness of breath"

        return self._first_patient_statement(ctx)

    def _hpi(self, ctx: Dict[str, Any]) -> str:
        """
        transcript 기반 HPI 구성.
        실제 의사 노트처럼 짧고 핵심적으로.
        """
        patient_txt = self._full_patient_text(ctx)
        profile = self._patient_profile(ctx)

        age = profile.get("age")
        sex = profile.get("sex") or profile.get("gender") or ""
        sex_word = "patient"
        if str(sex).upper().startswith("F"):
            sex_word = "female"
        elif str(sex).upper().startswith("M"):
            sex_word = "male"

        intro = "Patient"
        if age:
            intro = f"{age}-year-old {sex_word}"

        symptom_core = "symptoms"
        if any(p in patient_txt for p in ["epigastric", "upper stomach", "below my ribs", "stomach"]):
            if any(p in patient_txt for p in ["pressure", "heavy", "pressing down"]):
                symptom_core = "epigastric pressure-like pain"
            else:
                symptom_core = "epigastric pain"
        elif any(p in patient_txt for p in ["chest pain", "chest pressure", "chest tightness"]):
            symptom_core = "chest discomfort"

        features: List[str] = []
        if "shoulder" in patient_txt:
            features.append("radiation to the left shoulder")
        if any(p in patient_txt for p in ["nausea", "nauseous"]):
            features.append("associated nausea")
        if any(p in patient_txt for p in ["clammy", "cold sweat", "sweat", "diaphoresis"]):
            features.append("diaphoresis")
        if any(p in patient_txt for p in ["short of breath", "winded", "sob"]):
            features.append("mild shortness of breath")

        hpi = f"{intro} presents with {symptom_core}"

        if features:
            hpi += ", " + ", ".join(features)

        onset_bits: List[str] = []
        if any(p in patient_txt for p in ["3 or 4 hours", "three or four hours", "3-4 hours"]):
            onset_bits.append("starting about 3-4 hours prior to presentation")
        elif "today" in patient_txt:
            onset_bits.append("starting earlier today")

        if "after breakfast" in patient_txt:
            onset_bits.append("after breakfast")

        if onset_bits:
            hpi += ". Symptoms began " + ", ".join(onset_bits)

        if "tums" in patient_txt and any(
            p in patient_txt for p in ["hasn’t helped", "hasn't helped", "didn't help", "not helped", "without relief"]
        ):
            hpi += ". Symptoms were not relieved by antacids"

        if self._pmhx_list(ctx):
            hpi += f". PMH notable for {self._pmhx(ctx)}"

        if not hpi.endswith("."):
            hpi += "."

        return hpi

    def _focused_ros(self, ctx: Dict[str, Any]) -> str:
        txt = self._full_patient_text(ctx)

        positives: List[str] = []
        negatives: List[str] = []

        if any(p in txt for p in ["nausea", "nauseous"]):
            positives.append("positive for nausea")
        if any(p in txt for p in ["clammy", "cold sweat", "sweat", "diaphoresis"]):
            positives.append("positive for diaphoresis")
        if any(p in txt for p in ["short of breath", "winded", "sob"]):
            positives.append("positive for shortness of breath")
        if "shoulder" in txt:
            positives.append("positive for left shoulder radiation")

        if "vomit" not in txt and "vomiting" not in txt:
            negatives.append("negative for vomiting")

        if not positives and not negatives:
            return "Focused review of systems otherwise negative except as documented in HPI."

        parts = []
        if positives:
            parts.append(", ".join(positives))
        if negatives:
            parts.append(", ".join(negatives))

        return ". ".join(parts) + "."

    def _diagnostic_lines(self, ctx: Dict[str, Any]) -> List[str]:
        lines: List[str] = []
        crit = self._critical_labs(ctx)
        for lab in crit:
            name = lab.get("name") or "Lab"
            val = lab.get("value", "")
            unit = lab.get("unit", "")
            flag = lab.get("flag", "")
            line = f"{name}: {val}"
            if unit:
                line += f" {unit}"
            if flag:
                line += f" ({flag})"
            lines.append(line)
        return lines

    def _assessment_lines(self, ctx: Dict[str, Any], setting: str) -> List[str]:
        lines: List[str] = []
        acs = self._acs_concern(ctx)
        troponin_high = self._troponin_high(ctx)

        if acs:
            if troponin_high:
                lines.append("High concern for acute coronary syndrome / NSTEMI.")
            else:
                lines.append("Atypical acute coronary syndrome remains a significant concern.")
        else:
            lines.append("Acute symptoms under evaluation.")

        if self._has_phrase(ctx, ["epigastric", "stomach", "upper stomach"]):
            lines.append("Epigastric pain.")
        if self._has_phrase(ctx, ["nausea", "nauseous"]):
            lines.append("Nausea.")
        if self._has_phrase(ctx, ["short of breath", "winded"]):
            lines.append("Shortness of breath.")

        out: List[str] = []
        seen = set()
        for line in lines:
            if line not in seen:
                out.append(line)
                seen.add(line)

        return out[:3]

    def _mdm_text_uc(self, ctx: Dict[str, Any]) -> str:
        acs = self._acs_concern(ctx)
        troponin_high = self._troponin_high(ctx)
        pmhx = self._pmhx(ctx)

        if acs and troponin_high:
            return (
                "Given age, cardiovascular risk factors, symptom profile, and elevated troponin, "
                "acute coronary syndrome is a primary concern. Higher level evaluation is recommended."
            )

        if acs:
            return (
                "Given age, cardiovascular risk factors, and atypical symptom profile, acute coronary syndrome "
                "cannot be excluded in the urgent care setting."
            )

        return "Clinical presentation reviewed and further evaluation guided by history, exam, and available data."

    def _mdm_text_ed(self, ctx: Dict[str, Any]) -> str:
        acs = self._acs_concern(ctx)
        troponin_high = self._troponin_high(ctx)

        if acs and troponin_high:
            return (
                "Presentation is concerning for acute coronary syndrome / NSTEMI given risk factors, concerning symptoms, "
                "and elevated troponin. Ongoing cardiac evaluation with monitoring and specialty input is warranted."
            )

        if acs:
            return (
                "Presentation is concerning for atypical acute coronary syndrome and warrants continued cardiac evaluation, "
                "serial biomarkers, and repeat ECG as indicated."
            )

        return "Medical decision making based on current presentation, diagnostic data, and reassessment."

    def _plan_lines_uc(self, ctx: Dict[str, Any]) -> List[str]:
        lines: List[str] = []
        acs = self._acs_concern(ctx)

        orders = self._orders(ctx)
        if orders:
            lines.append("Recommend immediate ECG and cardiac enzyme evaluation.")

        if acs:
            lines.append("Recommend transfer to the emergency department / higher level of care for further evaluation and monitoring.")
            lines.append("Patient advised that cardiac etiology remains a significant concern.")
        else:
            lines.append("Continue diagnostic evaluation based on clinical course.")

        lines.append("Return/ED precautions reviewed for worsening pain, shortness of breath, syncope, or new concerning symptoms.")
        return lines

    def _plan_lines_ed(self, ctx: Dict[str, Any]) -> List[str]:
        lines: List[str] = []
        acs = self._acs_concern(ctx)
        troponin_high = self._troponin_high(ctx)

        if acs:
            lines.append("Continue cardiac evaluation with ECG, troponin trending, and monitoring.")
            if troponin_high:
                lines.append("Cardiology consultation and hospital admission / observation are recommended.")
            else:
                lines.append("Serial troponins and repeat ECG should be considered.")
            lines.append("Aspirin should be considered if no contraindication.")
        else:
            lines.append("Further diagnostic evaluation guided by clinical course and reassessment.")

        return lines

    def _disposition_uc(self, ctx: Dict[str, Any]) -> str:
        if self._acs_concern(ctx):
            return "Recommend immediate transfer to the emergency department for higher level evaluation."
        return "Disposition based on clinical reassessment and response to treatment."

    def _disposition_ed(self, ctx: Dict[str, Any]) -> str:
        if self._acs_concern(ctx):
            if self._troponin_high(ctx):
                return "Admission / observation recommended for ongoing cardiac evaluation."
            return "Further ED monitoring and repeat cardiac evaluation recommended."
        return "Disposition based on reassessment and diagnostic results."

    # -----------------------------
    # Urgent Care Template
    # -----------------------------

    def _compose_uc_note(self, ctx: Dict[str, Any]) -> str:
        cc = self._chief_complaint(ctx)
        hpi = self._hpi(ctx)
        ros = self._focused_ros(ctx)
        pmhx = self._pmhx(ctx)
        allergies = self._allergies(ctx)
        meds = self._meds(ctx)
        vitals = self._vitals(ctx)

        diagnostics = self._diagnostic_lines(ctx)
        assessment = self._assessment_lines(ctx, setting="uc")
        mdm = self._mdm_text_uc(ctx)
        plan = self._plan_lines_uc(ctx)
        disposition = self._disposition_uc(ctx)

        diagnostics_text = "\n".join([f"- {x}" for x in diagnostics]) if diagnostics else "- No critical diagnostic results available."
        assessment_text = "\n".join([f"- {x}" for x in assessment]) if assessment else "- Acute symptoms under evaluation."
        plan_text = "\n".join([f"- {x}" for x in plan]) if plan else "- Continue clinical reassessment."

        note = f"""
CHIEF COMPLAINT
{cc}

HPI
{hpi}

ROS
{ros}

PAST MEDICAL HISTORY
{pmhx}

MEDICATIONS
{meds}

ALLERGIES
{allergies}

VITAL SIGNS
{vitals}

PHYSICAL EXAM
General: Alert, conversant, no acute toxic appearance.
Cardiovascular: Regular rate and rhythm.
Respiratory: No acute respiratory distress.
Abdomen: Soft, no peritoneal signs on limited urgent care assessment.

DIAGNOSTICS
{diagnostics_text}

ASSESSMENT
{assessment_text}

MEDICAL DECISION MAKING
{mdm}

PLAN
{plan_text}

DISPOSITION
{disposition}
""".strip()

        return note

    # -----------------------------
    # Emergency Department Template
    # -----------------------------

    def _compose_ed_note(self, ctx: Dict[str, Any]) -> str:
        cc = self._chief_complaint(ctx)
        hpi = self._hpi(ctx)
        ros = self._focused_ros(ctx)
        pmhx = self._pmhx(ctx)
        vitals = self._vitals(ctx)
        allergies = self._allergies(ctx)
        meds = self._meds(ctx)

        diagnostics = self._diagnostic_lines(ctx)
        assessment = self._assessment_lines(ctx, setting="ed")
        mdm = self._mdm_text_ed(ctx)
        plan = self._plan_lines_ed(ctx)
        disposition = self._disposition_ed(ctx)

        diagnostics_text = "\n".join([f"- {x}" for x in diagnostics]) if diagnostics else "- No critical diagnostic results available."
        assessment_text = "\n".join([f"- {x}" for x in assessment]) if assessment else "- Acute symptoms under evaluation."
        plan_text = "\n".join([f"- {x}" for x in plan]) if plan else "- Continue reassessment."

        note = f"""
CHIEF COMPLAINT
{cc}

HISTORY OF PRESENT ILLNESS
{hpi}

REVIEW OF SYSTEMS
{ros}

PAST MEDICAL HISTORY
{pmhx}

MEDICATIONS
{meds}

ALLERGIES
{allergies}

VITAL SIGNS
{vitals}

PHYSICAL EXAM
General: Alert and interactive.
Cardiovascular: Regular rate and rhythm.
Respiratory: No acute respiratory distress.
Abdomen: Soft, no peritoneal signs on current exam.

DIAGNOSTIC RESULTS
{diagnostics_text}

IMPRESSION
{assessment_text}

MEDICAL DECISION MAKING
{mdm}

PLAN
{plan_text}

DISPOSITION
{disposition}
""".strip()

        return note