import os
import json
import sqlite3
import time
import uuid
from typing import Any, Dict, List, Optional


class NoteStore:
    """
    SQLite-backed audit/history store.

    역할:
    1) encounter 생성 / 종료
    2) note version 저장
    3) event log 저장
    4) 나중에 note history / events 조회 가능

    중요한 원칙:
    - 현재 화면의 "정식 현재 상태"는 여전히 hub.data 가 담당
    - SQLite는 append-only 기록 저장소 역할만 담당
    - 즉, DB가 깨져도 UI가 죽지 않도록 설계
    """

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            data_dir = os.path.join(root_dir, "data")
            os.makedirs(data_dir, exist_ok=True)
            db_path = os.path.join(data_dir, "safety_os.db")

        self.db_path = db_path
        self._init_db()

    # -----------------------------
    # Internal helpers
    # -----------------------------
    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        conn = self._connect()
        try:
            cur = conn.cursor()

            # encounter 1회 단위
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS encounters (
                    encounter_id TEXT PRIMARY KEY,
                    patient_id TEXT,
                    note_style TEXT,
                    status TEXT,
                    started_at REAL,
                    ended_at REAL
                )
                """
            )

            # note 버전 기록
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS note_versions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    encounter_id TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    source TEXT NOT NULL,
                    reason TEXT,
                    note_text TEXT NOT NULL,
                    meta_json TEXT,
                    FOREIGN KEY(encounter_id) REFERENCES encounters(encounter_id)
                )
                """
            )

            # event/audit log
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS event_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    encounter_id TEXT,
                    created_at REAL NOT NULL,
                    event_type TEXT NOT NULL,
                    payload_json TEXT,
                    FOREIGN KEY(encounter_id) REFERENCES encounters(encounter_id)
                )
                """
            )

            # 조회 성능용 인덱스
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_note_versions_encounter_id ON note_versions(encounter_id)"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_event_log_encounter_id ON event_log(encounter_id)"
            )

            conn.commit()
        finally:
            conn.close()

    def _to_json(self, value: Any) -> str:
        try:
            return json.dumps(value, ensure_ascii=False)
        except Exception:
            return json.dumps({"repr": str(value)}, ensure_ascii=False)

    # -----------------------------
    # Encounter methods
    # -----------------------------
    def create_encounter(
        self,
        patient_id: Optional[str] = None,
        note_style: str = "uc",
    ) -> str:
        """
        새로운 encounter 생성.
        encounter_id 반환.
        """
        encounter_id = f"enc-{int(time.time())}-{uuid.uuid4().hex[:8]}"

        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO encounters (
                    encounter_id, patient_id, note_style, status, started_at, ended_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    encounter_id,
                    patient_id,
                    note_style,
                    "active",
                    time.time(),
                    None,
                ),
            )
            conn.commit()
        finally:
            conn.close()

        return encounter_id

    def close_encounter(self, encounter_id: str) -> None:
        """
        encounter 종료 처리
        """
        if not encounter_id:
            return

        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                UPDATE encounters
                SET status = ?, ended_at = ?
                WHERE encounter_id = ?
                """,
                ("closed", time.time(), encounter_id),
            )
            conn.commit()
        finally:
            conn.close()

    def get_encounter(self, encounter_id: str) -> Optional[Dict[str, Any]]:
        if not encounter_id:
            return None

        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT * FROM encounters WHERE encounter_id = ?",
                (encounter_id,),
            )
            row = cur.fetchone()
            if not row:
                return None
            return dict(row)
        finally:
            conn.close()

    # -----------------------------
    # Note version methods
    # -----------------------------
    def append_note_version(
        self,
        encounter_id: str,
        note_text: str,
        source: str,
        reason: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        노트 버전 1개 저장.
        source 예:
        - ai_doctor
        - user_edit
        - note_engine
        - reset
        """
        if not encounter_id:
            return

        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO note_versions (
                    encounter_id, created_at, source, reason, note_text, meta_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    encounter_id,
                    time.time(),
                    source,
                    reason,
                    note_text or "",
                    self._to_json(meta or {}),
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def get_note_versions(self, encounter_id: str) -> List[Dict[str, Any]]:
        if not encounter_id:
            return []

        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT * FROM note_versions
                WHERE encounter_id = ?
                ORDER BY created_at ASC, id ASC
                """,
                (encounter_id,),
            )
            rows = cur.fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def get_latest_note(self, encounter_id: str) -> Optional[Dict[str, Any]]:
        if not encounter_id:
            return None

        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT * FROM note_versions
                WHERE encounter_id = ?
                ORDER BY created_at DESC, id DESC
                LIMIT 1
                """,
                (encounter_id,),
            )
            row = cur.fetchone()
            if not row:
                return None
            return dict(row)
        finally:
            conn.close()

    # -----------------------------
    # Event log methods
    # -----------------------------
    def append_event(
        self,
        event_type: str,
        payload: Optional[Dict[str, Any]] = None,
        encounter_id: Optional[str] = None,
    ) -> None:
        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO event_log (
                    encounter_id, created_at, event_type, payload_json
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    encounter_id,
                    time.time(),
                    event_type,
                    self._to_json(payload or {}),
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def get_events(self, encounter_id: str) -> List[Dict[str, Any]]:
        if not encounter_id:
            return []

        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT * FROM event_log
                WHERE encounter_id = ?
                ORDER BY created_at ASC, id ASC
                """,
                (encounter_id,),
            )
            rows = cur.fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()