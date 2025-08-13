"""
Модуль для хранения и управления данными пользователей (SQLite, aiosqlite).
"""

import os
import aiosqlite
import time
from typing import Dict, List, Any, Optional, Tuple
from config import logger, ENABLE_DIALOG_LOGGING


DATA_DIR = "data"
DB_PATH = os.path.join(DATA_DIR, "bot.db")
os.makedirs(DATA_DIR, exist_ok=True)


class Storage:
    """
    Хранилище на aiosqlite: пользователи, сообщения, напоминания.
    """

    def __init__(self) -> None:
        self._initialized: bool = False

    async def _init_schema(self) -> None:
        if self._initialized:
            return
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    user_id INTEGER PRIMARY KEY,
                    chat_id INTEGER
                )
                """
            )
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    ts REAL NOT NULL
                )
                """
            )
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS reminders (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    time_hhmm TEXT NOT NULL,
                    text TEXT NOT NULL,
                    last_sent_date TEXT,
                    date_once TEXT,
                    weekdays TEXT,
                    weekday_only INTEGER DEFAULT 0,
                    weekend_only INTEGER DEFAULT 0,
                    snooze_until TEXT
                )
                """
            )
            await db.commit()
        self._initialized = True

    # ===== utils =====
    async def _ensure_user_row(self, user_id: int) -> None:
        await self._init_schema()
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute("INSERT OR IGNORE INTO users(user_id) VALUES (?)", (user_id,))
            await db.commit()

    # ===== users/chat =====
    async def set_chat_id(self, user_id: int, chat_id: int) -> None:
        await self._ensure_user_row(user_id)
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute(
                "INSERT INTO users(user_id, chat_id) VALUES (?, ?)\n"
                "ON CONFLICT(user_id) DO UPDATE SET chat_id=excluded.chat_id",
                (user_id, int(chat_id)),
            )
            await db.commit()

    async def get_chat_id(self, user_id: int) -> Optional[int]:
        await self._init_schema()
        async with aiosqlite.connect(DB_PATH) as db:
            async with db.execute("SELECT chat_id FROM users WHERE user_id=?", (user_id,)) as cur:
                row = await cur.fetchone()
                return int(row[0]) if row and row[0] is not None else None

    # ===== messages/history =====
    async def add_message(self, user_id: int, role: str, content: str) -> None:
        if not ENABLE_DIALOG_LOGGING:
            return
        await self._ensure_user_row(user_id)
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute(
                "INSERT INTO messages(user_id, role, content, ts) VALUES (?, ?, ?, ?)",
                (user_id, role, content, time.time()),
            )
            await db.commit()

    async def get_message_history(self, user_id: int, max_messages: int = 50) -> List[Dict[str, str]]:
        await self._init_schema()
        async with aiosqlite.connect(DB_PATH) as db:
            async with db.execute(
                "SELECT role, content FROM messages WHERE user_id=? ORDER BY id DESC LIMIT ?",
                (user_id, max_messages),
            ) as cur:
                rows = await cur.fetchall()
        result = [{"role": r[0], "content": r[1]} for r in rows][::-1]
        return result

    async def clear_history(self, user_id: int) -> None:
        await self._init_schema()
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute("DELETE FROM messages WHERE user_id=?", (user_id,))
            await db.commit()

    # ===== reminders =====
    async def add_reminder(
        self,
        user_id: int,
        time_hh_mm: str,
        text: str,
        *,
        date_once: Optional[str] = None,
        weekdays: Optional[List[int]] = None,
        weekday_only: bool = False,
        weekend_only: bool = False,
    ) -> Dict[str, Any]:
        # Валидация времени
        try:
            hours_str, minutes_str = time_hh_mm.split(":")
            hours = int(hours_str)
            minutes = int(minutes_str)
            if not (0 <= hours <= 23 and 0 <= minutes <= 59):
                raise ValueError
        except Exception:
            raise ValueError("Неверный формат времени. Используйте HH:MM, например 09:30")

        await self._ensure_user_row(user_id)
        weekdays_str = None
        if weekdays:
            weekdays_sorted = sorted(int(d) for d in weekdays)
            weekdays_str = ",".join(str(d) for d in weekdays_sorted)

        async with aiosqlite.connect(DB_PATH) as db:
            cur = await db.execute(
                """
                INSERT INTO reminders(user_id, time_hhmm, text, last_sent_date, date_once, weekdays, weekday_only, weekend_only, snooze_until)
                VALUES (?, ?, ?, NULL, ?, ?, ?, ?, NULL)
                """,
                (
                    user_id,
                    f"{hours:02d}:{minutes:02d}",
                    text.strip(),
                    date_once,
                    weekdays_str,
                    1 if weekday_only else 0,
                    1 if weekend_only else 0,
                ),
            )
            await db.commit()
            reminder_id = cur.lastrowid
        return {
            "id": int(reminder_id),
            "time": f"{hours:02d}:{minutes:02d}",
            "text": text.strip(),
        }

    async def list_reminders(self, user_id: int) -> List[Dict[str, Any]]:
        await self._init_schema()
        async with aiosqlite.connect(DB_PATH) as db:
            async with db.execute(
                "SELECT id, time_hhmm, text FROM reminders WHERE user_id=? ORDER BY time_hhmm ASC, id ASC",
                (user_id,),
            ) as cur:
                rows = await cur.fetchall()
        return [
            {"id": int(r[0]), "time": r[1], "text": r[2]}
            for r in rows
        ]

    async def delete_reminder(self, user_id: int, identifier: int) -> bool:
        await self._init_schema()
        async with aiosqlite.connect(DB_PATH) as db:
            cur = await db.execute(
                "DELETE FROM reminders WHERE user_id=? AND id=?",
                (user_id, int(identifier)),
            )
            await db.commit()
            if cur.rowcount and cur.rowcount > 0:
                return True

        items = await self.list_reminders(user_id)
        index_zero_based = int(identifier) - 1
        if 0 <= index_zero_based < len(items):
            real_id = items[index_zero_based]["id"]
            async with aiosqlite.connect(DB_PATH) as db:
                await db.execute(
                    "DELETE FROM reminders WHERE user_id=? AND id=?",
                    (user_id, int(real_id)),
                )
                await db.commit()
            return True
        return False

    async def get_due_reminders(
        self,
        time_hh_mm: str,
        today_date: str,
        weekday_idx: int,
        now_iso: str,
    ) -> List[Tuple[int, Dict[str, Any], bool]]:
        """
        Возвращает список напоминаний, которые нужно отправить сейчас.
        Учитывает два источника срабатывания:
        - нормальное время (time_hhmm == текущему) + условия расписания
        - отложенное срабатывание (snooze_until <= now_iso)

        Возвращает кортежи (user_id, reminder_row_as_dict, is_snooze).
        При is_snooze=True нужно только сбросить snooze, last_sent_date менять не надо.
        При обычном срабатывании — установить last_sent_date=today_date и сбросить snooze.
        """
        await self._init_schema()
        due: List[Tuple[int, Dict[str, Any], bool]] = []

        async with aiosqlite.connect(DB_PATH) as db:
            # Напоминания по основному времени (не snooze), ещё не отправленные сегодня
            async with db.execute(
                """
                SELECT id, user_id, time_hhmm, text, last_sent_date, date_once, weekdays, weekday_only, weekend_only, snooze_until
                FROM reminders
                WHERE time_hhmm=?
                  AND (last_sent_date IS NULL OR last_sent_date<>?)
                """,
                (time_hh_mm, today_date),
            ) as cur:
                rows_time = await cur.fetchall()

        for r in rows_time:
            rid, uid, time_val, text_val, last_sent, date_once, weekdays_str, weekday_only, weekend_only, snooze_until = r

            # Условия расписания
            is_weekday = 0 <= weekday_idx <= 4
            is_weekend = 5 <= weekday_idx <= 6

            if date_once and date_once != today_date:
                continue
            if weekday_only and not is_weekday:
                continue
            if weekend_only and not is_weekend:
                continue
            if weekdays_str:
                try:
                    allowed = {int(x) for x in weekdays_str.split(',') if x}
                except Exception:
                    allowed = set()
                if weekday_idx not in allowed:
                    continue

            due.append(
                (int(uid), {"id": int(rid), "time": time_val, "text": text_val}, False)
            )

        # Срабатывания по snooze (игнорируют last_sent_date)
        async with aiosqlite.connect(DB_PATH) as db:
            async with db.execute(
                """
                SELECT id, user_id, time_hhmm, text, snooze_until
                FROM reminders
                WHERE snooze_until IS NOT NULL AND snooze_until<=?
                """,
                (now_iso,),
            ) as cur2:
                rows_snooze = await cur2.fetchall()
        for r in rows_snooze:
            rid, uid, time_val, text_val, snooze_until = r
            due.append(
                (int(uid), {"id": int(rid), "time": time_val, "text": text_val}, True)
            )

        return due

    async def mark_reminder_sent(self, user_id: int, reminder_id: int, today_date: str) -> None:
        await self._init_schema()
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute(
                "UPDATE reminders SET last_sent_date=?, snooze_until=NULL WHERE user_id=? AND id=?",
                (today_date, user_id, int(reminder_id)),
            )
            await db.commit()

    async def clear_snooze(self, user_id: int, reminder_id: int) -> None:
        await self._init_schema()
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute(
                "UPDATE reminders SET snooze_until=NULL WHERE user_id=? AND id=?",
                (user_id, int(reminder_id)),
            )
            await db.commit()

    async def set_reminder_snooze(self, user_id: int, reminder_id: int, snooze_until_iso: str) -> None:
        await self._init_schema()
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute(
                "UPDATE reminders SET snooze_until=? WHERE user_id=? AND id=?",
                (snooze_until_iso, user_id, int(reminder_id)),
            )
            await db.commit()


storage = Storage()