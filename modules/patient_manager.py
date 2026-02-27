"""
Patient Manager

- Demo용 환자(EMR) 스냅샷 제공
- 실제 구현 시: EMR 연동/DB/HL7/FHIR 연결 지점
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Optional


class PatientManager:
    def __init__(self, *, patients_db_path: Optional[str] = None):
        self.patients_db_path = patients_db_path

        # fallback mock DB (파일이 없거나 포맷이 틀려도 앱이 죽지 않도록)
        self._mock_db: Dict[str, Dict[str, Any]] = {
            "P1001": {
                "patient_id": "P1001",
                "profile": {"name": "Jane Doe", "age": 73, "sex": "F"},
                "problems": ["Type 2 DM", "HTN", "Hyperlipidemia"],
                "allergies": ["Penicillin (rash)"],
                "medications": ["Metformin", "Lisinopril", "Aspirin"],
                "history_summary": "DM/HTN/HLD. No known CAD documented.",
                # ✅ 기본 vitals(데모). 실제로는 patients_db.json 값이 있으면 그게 우선
                "current_vitals": {"bp": "140/90", "hr": "88", "rr": "18", "spo2": "97%", "temp": "98.4F"},
                "historical_labs": {
                    "Troponin-I": [
                        {"dt": "2023-08-15 09:00", "value": 0.01, "unit": "ng/mL", "ref_range": "< 0.04"},
                        {"dt": "2024-01-20 08:10", "value": 0.01, "unit": "ng/mL", "ref_range": "< 0.04"},
                    ],
                    "Creatinine": [
                        {"dt": "2023-08-15 09:00", "value": 0.8, "unit": "mg/dL", "ref_range": "0.6 - 1.2"},
                        {"dt": "2024-01-20 08:10", "value": 0.9, "unit": "mg/dL", "ref_range": "0.6 - 1.2"},
                    ],
                },
            }
        }

        self._json_db: Dict[str, Any] = {}
        self._load_json_db()

    def _load_json_db(self) -> None:
        """
        patients_db.json 위치 후보들을 순서대로 탐색해서 하나라도 찾으면 로딩합니다.
        """
        candidates = []
        if self.patients_db_path:
            candidates.append(self.patients_db_path)

        candidates.extend(
            [
                os.path.join(os.getcwd(), "patients_db.json"),
                os.path.join(os.path.dirname(os.path.dirname(__file__)), "patients_db.json"),
                os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "patients_db.json"),
            ]
        )

        for path in candidates:
            try:
                if path and os.path.exists(path):
                    with open(path, "r", encoding="utf-8") as f:
                        loaded = json.load(f)
                    self._json_db = loaded if isinstance(loaded, dict) else {}
                    return
            except Exception:
                continue

        # 못 찾으면 빈 dict 유지
        self._json_db = {}

    def list_patient_ids(self):
        ids = set(self._mock_db.keys()) | set(self._json_db.keys())
        return sorted(ids)

    def _coerce_float_from_value_str(self, value_str: str) -> Optional[float]:
        """
        "0.01 ng/mL" 같은 문자열에서 숫자만 뽑아 float로 변환 시도
        """
        m = re.search(r"-?\d+(?:\.\d+)?", value_str)
        if not m:
            return None
        try:
            return float(m.group(0))
        except Exception:
            return None

    def get_patient_snapshot(self, patient_id: str) -> Optional[Dict[str, Any]]:
        """
        반환 형태(권장):
        {
          "patient_id": "P1001",
          "profile": {"name":..., "age":..., "sex":...},
          "problems": [...],
          "allergies": [...],
          "medications": [...],
          "current_vitals": {...},
          "historical_labs": {...}
        }
        """
        # 1) JSON DB 우선
        if patient_id in self._json_db:
            raw = self._json_db[patient_id]
            p: Dict[str, Any] = dict(raw) if isinstance(raw, dict) else {}

            # --- profile normalize ---
            prof = p.get("profile") if isinstance(p.get("profile"), dict) else {}

            # gender -> sex
            if "gender" in prof and "sex" not in prof:
                prof["sex"] = prof.get("gender")

            # JSON에서 pmhx/allergies/medications가 profile 안에 있는 구조도 지원
            if "pmhx" in prof and "problems" not in p:
                p["problems"] = prof.get("pmhx")
            if "allergies" in prof and "allergies" not in p:
                p["allergies"] = prof.get("allergies")
            if "medications" in prof and "medications" not in p:
                p["medications"] = prof.get("medications")

            p["profile"] = prof

            # --- ✅ current_vitals normalize (핵심) ---
            # 권장: 최상위 "current_vitals"
            # 허용: profile 안 "current_vitals"
            # 허용: 최상위 "vitals" (혹시 다른 이름으로 저장했을 때)
            cv = None
            if isinstance(p.get("current_vitals"), dict):
                cv = p.get("current_vitals")
            elif isinstance(prof.get("current_vitals"), dict):
                cv = prof.get("current_vitals")
            elif isinstance(p.get("vitals"), dict):
                cv = p.get("vitals")

            p["current_vitals"] = cv if isinstance(cv, dict) else {}

            # --- historical_labs normalize ---
            # (1) dt/value/unit/ref_range 구조
            # (2) date/value(string)/unit 구조
            h = p.get("historical_labs")
            if isinstance(h, dict):
                normalized_h = {}
                for lab, rows in h.items():
                    if not isinstance(rows, list):
                        continue
                    out_rows = []
                    for r in rows:
                        if not isinstance(r, dict):
                            continue
                        rr = dict(r)
                        if "dt" not in rr and "date" in rr:
                            rr["dt"] = rr.get("date")

                        # value가 "0.01 ng/mL" 같은 문자열이면 숫자로 변환 시도
                        if isinstance(rr.get("value"), str):
                            maybe = self._coerce_float_from_value_str(rr["value"])
                            if maybe is not None:
                                rr["value"] = maybe

                        out_rows.append(rr)

                    if out_rows:
                        normalized_h[lab] = out_rows
                p["historical_labs"] = normalized_h
            else:
                p["historical_labs"] = {}

            # --- patient_id ensure ---
            if "patient_id" not in p:
                p["patient_id"] = patient_id

            # --- problems/allergies/medications 타입 안전장치 ---
            for key in ["problems", "allergies", "medications"]:
                if key not in p or p[key] is None:
                    p[key] = []
                if not isinstance(p[key], list):
                    p[key] = [str(p[key])]

            return p

        # 2) fallback mock DB
        if patient_id in self._mock_db:
            return self._mock_db[patient_id]

        return None