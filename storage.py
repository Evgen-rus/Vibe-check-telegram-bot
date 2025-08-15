"""
Модуль для хранения и управления данными пользователей (SQLite, aiosqlite).
"""

import os
import aiosqlite
import time
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from config import logger, ENABLE_MESSAGE_HISTORY, LOCAL_TZ


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
                CREATE TABLE IF NOT EXISTS user_profiles (
                    user_id INTEGER PRIMARY KEY,
                    sex TEXT,
                    age INTEGER,
                    height_cm INTEGER,
                    weight_kg REAL,
                    activity TEXT,
                    goal TEXT,
                    allergies TEXT,
                    diet TEXT
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
            # Добавляем недостающие колонки для периодических напоминаний
            try:
                cols: List[str] = []
                async with db.execute("PRAGMA table_info(reminders)") as cinfo:
                    rows = await cinfo.fetchall()
                    cols = [str(r[1]) for r in rows]
                async def add_col(name: str, ddl: str) -> None:
                    await db.execute(f"ALTER TABLE reminders ADD COLUMN {name} {ddl}")
                if "period_minutes" not in cols:
                    await add_col("period_minutes", "INTEGER")
                if "window_start_hhmm" not in cols:
                    await add_col("window_start_hhmm", "TEXT")
                if "window_end_hhmm" not in cols:
                    await add_col("window_end_hhmm", "TEXT")
                if "next_fire_at" not in cols:
                    await add_col("next_fire_at", "TEXT")
            except Exception:
                pass
            await db.commit()
        self._initialized = True

    # ===== utils =====
    async def _ensure_user_row(self, user_id: int) -> None:
        await self._init_schema()
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute("INSERT OR IGNORE INTO users(user_id) VALUES (?)", (user_id,))
            await db.execute("INSERT OR IGNORE INTO user_profiles(user_id) VALUES (?)", (user_id,))
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

    # ===== user profile =====
    async def get_profile(self, user_id: int) -> Dict[str, Any]:
        await self._ensure_user_row(user_id)
        async with aiosqlite.connect(DB_PATH) as db:
            async with db.execute(
                "SELECT sex, age, height_cm, weight_kg, activity, goal, allergies, diet FROM user_profiles WHERE user_id=?",
                (user_id,),
            ) as cur:
                row = await cur.fetchone()
        if not row:
            return {
                "sex": None,
                "age": None,
                "height_cm": None,
                "weight_kg": None,
                "activity": None,
                "goal": None,
                "allergies": None,
                "diet": None,
            }
        return {
            "sex": row[0],
            "age": row[1],
            "height_cm": row[2],
            "weight_kg": row[3],
            "activity": row[4],
            "goal": row[5],
            "allergies": row[6],
            "diet": row[7],
        }

    async def set_profile_fields(self, user_id: int, **fields: Any) -> None:
        if not fields:
            return
        await self._ensure_user_row(user_id)
        allowed = {"sex", "age", "height_cm", "weight_kg", "activity", "goal", "allergies", "diet"}
        updates = {k: v for k, v in fields.items() if k in allowed}
        if not updates:
            return
        cols = ", ".join(f"{k}=?" for k in updates.keys())
        params = list(updates.values()) + [user_id]
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute(f"UPDATE user_profiles SET {cols} WHERE user_id=?", params)
            await db.commit()

    async def get_compact_profile_context(self, user_id: int) -> str:
        prof = await self.get_profile(user_id)
        parts: List[str] = []
        sex_map = {"m": "мужчина", "f": "женщина"}
        if prof.get("sex"):
            parts.append(f"пол: {sex_map.get(prof['sex'], prof['sex'])}")
        if prof.get("age") is not None:
            parts.append(f"возраст: {int(prof['age'])} лет")
        if prof.get("height_cm") is not None:
            parts.append(f"рост: {int(prof['height_cm'])} см")
        if prof.get("weight_kg") is not None:
            try:
                w = float(prof["weight_kg"])
                parts.append(f"вес: {w:.1f} кг")
            except Exception:
                parts.append(f"вес: {prof['weight_kg']}")
        if prof.get("activity"):
            act_map = {"low": "низкая", "medium": "средняя", "high": "высокая"}
            parts.append(f"активность: {act_map.get(prof['activity'], prof['activity'])}")
        if prof.get("goal"):
            goal_map = {"lose": "снижение веса", "maintain": "поддержание", "gain": "набор веса"}
            parts.append(f"цель: {goal_map.get(prof['goal'], prof['goal'])}")
        if prof.get("allergies"):
            parts.append(f"аллергии: {prof['allergies']}")
        if prof.get("diet"):
            parts.append(f"диета/ограничения: {prof['diet']}")
        return "; ".join(parts)

    # ===== messages/history =====
    async def add_message(self, user_id: int, role: str, content: str) -> None:
        if not ENABLE_MESSAGE_HISTORY:
            return
        await self._ensure_user_row(user_id)
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute(
                "INSERT INTO messages(user_id, role, content, ts) VALUES (?, ?, ?, ?)",
                (user_id, role, content, time.time()),
            )
            await db.commit()
    
    async def get_message_history(self, user_id: int, max_messages: int = 20) -> List[Dict[str, str]]:
        """
        Получает историю сообщений пользователя.
        max_messages - количество сообщений в истории для модели.
        Порядок: берутся последние max_messages из БД по убыванию id, 
        затем список разворачивается — на выходе хронологический порядок (от старых к новым).
        """
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
        period_minutes: Optional[int] = None,
        window_start_hhmm: Optional[str] = None,
        window_end_hhmm: Optional[str] = None,
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

        # Рассчитываем next_fire_at для периодических напоминаний
        next_fire_at: Optional[str] = None
        if isinstance(period_minutes, int) and period_minutes > 0:
            now_local = datetime.now(LOCAL_TZ)
            fire = now_local + timedelta(minutes=int(period_minutes))

            def is_day_allowed(dt):
                idx = dt.weekday()
                if weekday_only and idx > 4:
                    return False
                if weekend_only and idx < 5:
                    return False
                if weekdays:
                    try:
                        return idx in {int(x) for x in weekdays}
                    except Exception:
                        return True
                return True

            def window_bounds(dt):
                if not (window_start_hhmm and window_end_hhmm):
                    return None
                try:
                    sh, sm = [int(x) for x in window_start_hhmm.split(":")]
                    eh, em = [int(x) for x in window_end_hhmm.split(":")]
                    start = dt.replace(hour=sh, minute=sm, second=0, microsecond=0)
                    end = dt.replace(hour=eh, minute=em, second=0, microsecond=0)
                    return (start, end)
                except Exception:
                    return None

            max_steps = 365
            steps = 0
            while True:
                steps += 1
                if steps > max_steps:
                    break
                if not is_day_allowed(fire):
                    fire = (fire.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1))
                wb = window_bounds(fire)
                if wb:
                    ws, we = wb
                    if fire < ws:
                        fire = ws
                    elif fire > we:
                        fire = ws + timedelta(days=1)
                        continue
                if (not wb or (wb and wb[0] <= fire <= wb[1])) and is_day_allowed(fire):
                    break
                fire = fire + timedelta(minutes=int(period_minutes))

            next_fire_at = fire.strftime("%Y-%m-%d %H:%M")

        async with aiosqlite.connect(DB_PATH) as db:
            cur = await db.execute(
                """
                INSERT INTO reminders(
                    user_id, time_hhmm, text, last_sent_date, date_once, weekdays, weekday_only, weekend_only,
                    snooze_until, period_minutes, window_start_hhmm, window_end_hhmm, next_fire_at
                )
                VALUES (?, ?, ?, NULL, ?, ?, ?, ?, NULL, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    f"{hours:02d}:{minutes:02d}",
                    text.strip(),
                    date_once,
                    weekdays_str,
                    1 if weekday_only else 0,
                    1 if weekend_only else 0,
                    period_minutes if period_minutes else None,
                    window_start_hhmm,
                    window_end_hhmm,
                    next_fire_at,
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
                """
                SELECT id, time_hhmm, text, date_once, weekdays, weekday_only, weekend_only, snooze_until,
                       period_minutes, window_start_hhmm, window_end_hhmm
                FROM reminders
                WHERE user_id=?
                ORDER BY time_hhmm ASC, id ASC
                """,
                (user_id,),
            ) as cur:
                rows = await cur.fetchall()
        result: List[Dict[str, Any]] = []
        for r in rows:
            result.append({
                "id": int(r[0]),
                "time": r[1],
                "text": r[2],
                "date_once": r[3],
                "weekdays": r[4],
                "weekday_only": bool(r[5]) if r[5] is not None else False,
                "weekend_only": bool(r[6]) if r[6] is not None else False,
                "snooze_until": r[7],
                "period_minutes": r[8],
                "window_start_hhmm": r[9],
                "window_end_hhmm": r[10],
            })
        return result

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
                (int(uid), {"id": int(rid), "time": time_val, "text": text_val, "periodic": None}, True)
            )

        # Периодические срабатывания по next_fire_at
        async with aiosqlite.connect(DB_PATH) as db:
            async with db.execute(
                """
                SELECT id, user_id, text
                FROM reminders
                WHERE period_minutes IS NOT NULL AND next_fire_at IS NOT NULL AND next_fire_at<=?
                """,
                (now_iso,),
            ) as cur3:
                rows_period = await cur3.fetchall()
        for r in rows_period:
            rid, uid, text_val = r
            due.append((int(uid), {"id": int(rid), "time": None, "text": text_val, "periodic": True}, False))

        return due

    async def get_compact_reminders_context(self, user_id: int, limit: int = 5) -> List[str]:
        """
        Возвращает краткий список ближайших напоминаний пользователя
        в виде строк: HH:MM — текст (c пометками расписания/даты).
        """
        items = await self.list_reminders(user_id)
        # Сортируем по времени и id
        items_sorted = sorted(items, key=lambda r: (r.get("time", ""), r.get("id", 0)))
        lines: List[str] = []
        for r in items_sorted[:limit]:
            extras = []
            # Метки расписания
            if r.get("date_once"):
                extras.append(f"дата {r['date_once']}")
            if r.get("weekday_only"):
                extras.append("будни")
            if r.get("weekend_only"):
                extras.append("выходные")
            if r.get("weekdays"):
                idx_to_ru = {0:"пн",1:"вт",2:"ср",3:"чт",4:"пт",5:"сб",6:"вс"}
                try:
                    items_days = [idx_to_ru.get(int(x), str(x)) for x in r['weekdays'].split(',') if x]
                    if items_days:
                        extras.append("дни: " + ",".join(items_days))
                except Exception:
                    pass
            # Форматирование заголовка
            title: str
            per = r.get("period_minutes")
            if per:
                wnd_start = r.get("window_start_hhmm")
                wnd_end = r.get("window_end_hhmm")
                wnd = f" {wnd_start}-{wnd_end}" if (wnd_start and wnd_end) else ""
                title = f"каждые {int(per)}мин{wnd}"
            else:
                title = r.get("time") or ""
            extra_str = f" ({'; '.join(extras)})" if extras else ""
            lines.append(f"{title} — {r['text']}{extra_str}")
        return lines

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

    async def bump_periodic_next_fire(self, user_id: int, reminder_id: int, now_iso: str) -> None:
        """
        Сдвигает next_fire_at вперёд на один интервал с учётом окна и ограничений по дням.
        """
        await self._init_schema()
        async with aiosqlite.connect(DB_PATH) as db:
            async with db.execute(
                "SELECT period_minutes, window_start_hhmm, window_end_hhmm, weekday_only, weekend_only, weekdays FROM reminders WHERE user_id=? AND id=?",
                (user_id, int(reminder_id)),
            ) as cur:
                row = await cur.fetchone()
            if not row:
                return
            per_min, wnd_start, wnd_end, wk_only, we_only, wds = row
            if not per_min:
                return
            base = datetime.strptime(now_iso, "%Y-%m-%d %H:%M")
            base = LOCAL_TZ.localize(base)
            fire = base + timedelta(minutes=int(per_min))

            def is_day_allowed(dt: datetime) -> bool:
                idx = dt.weekday()
                if wk_only and idx > 4:
                    return False
                if we_only and idx < 5:
                    return False
                if wds:
                    try:
                        return idx in {int(x) for x in str(wds).split(',') if x}
                    except Exception:
                        return True
                return True

            def window_bounds(dt: datetime) -> Optional[Tuple[datetime, datetime]]:
                if not (wnd_start and wnd_end):
                    return None
                try:
                    sh, sm = [int(x) for x in str(wnd_start).split(":")]
                    eh, em = [int(x) for x in str(wnd_end).split(":")]
                    start = dt.replace(hour=sh, minute=sm, second=0, microsecond=0)
                    end = dt.replace(hour=eh, minute=em, second=0, microsecond=0)
                    return (start, end)
                except Exception:
                    return None

            max_steps = 365
            steps = 0
            while True:
                steps += 1
                if steps > max_steps:
                    break
                if not is_day_allowed(fire):
                    fire = (fire.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1))
                wb = window_bounds(fire)
                if wb:
                    ws, we = wb
                    if fire < ws:
                        fire = ws
                    elif fire > we:
                        fire = ws + timedelta(days=1)
                        continue
                if (not wb or (wb and wb[0] <= fire <= wb[1])) and is_day_allowed(fire):
                    break
                fire = fire + timedelta(minutes=int(per_min))

            await db.execute(
                "UPDATE reminders SET next_fire_at=?, snooze_until=NULL WHERE user_id=? AND id=?",
                (fire.strftime("%Y-%m-%d %H:%M"), user_id, int(reminder_id)),
            )
            await db.commit()


storage = Storage()