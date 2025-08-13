"""
Vibe Checker - персональный помощник по питанию.
Использует OpenAI API для понимания естественной речи и поддерживает голосовые сообщения.

Usage:
    python main.py
"""

import asyncio
import io
import contextlib
from typing import List, Optional
from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command, BaseFilter
from aiogram.types import Message, BotCommand, InlineKeyboardMarkup, InlineKeyboardButton, CallbackQuery
from aiogram.exceptions import TelegramBadRequest

from config import TELEGRAM_BOT_TOKEN, logger, ALLOWED_USERS, LOCAL_TZ
from openai_module import get_vibe_checker_response
from storage import storage
from prompts import WELCOME_MESSAGE, HELP_MESSAGE
from audio_handler import transcribe_voice
from datetime import datetime, timedelta
import re
import time

# Инициализация бота и диспетчера
bot = Bot(token=TELEGRAM_BOT_TOKEN)
dp = Dispatcher()


class AccessFilter(BaseFilter):
    """
    Глобальный фильтр доступа: пропускает только пользователей из ALLOWED_USERS.
    Если пользователь не разрешен — отправляет уведомление и блокирует обработчик.
    """

    async def __call__(self, message: Message) -> bool:
        user_id = message.from_user.id if message.from_user else None
        if user_id is None or user_id not in ALLOWED_USERS:
            try:
                await message.answer("🚫 У вас нет доступа к этому боту.")
            except Exception as e:
                logger.error(f"Не удалось отправить сообщение об отказе в доступе: {e}")
            logger.warning(
                f"Попытка доступа от неавторизованного пользователя: {user_id}"
            )
            return False
        return True


dp.message.filter(AccessFilter())


def split_text_for_telegram(text: str, limit: int = 4096) -> List[str]:
    """
    Безопасно разбивает длинный текст на части <= limit, стараясь резать по абзацам, строкам и словам.
    """
    if len(text) <= limit:
        return [text]

    chunks: List[str] = []
    current: str = ""

    paragraphs = text.split("\n\n")
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            if current and len(current) + 2 <= limit:
                current += "\n\n"
            elif current:
                chunks.append(current)
                current = ""
            continue

        if not current and len(paragraph) <= limit:
            current = paragraph
            continue

        if len(current) + 2 + len(paragraph) <= limit:
            current = f"{current}\n\n{paragraph}" if current else paragraph
            continue

        # Параграф не влезает — режем по строкам/словам
        lines = paragraph.split("\n")
        for line in lines:
            line = line.rstrip()
            if not current:
                if len(line) <= limit:
                    current = line
                else:
                    # Режем строку по словам, в крайнем случае — по символам
                    part = ""
                    for word in line.split(" "):
                        candidate = (part + " " + word).strip() if part else word
                        if len(candidate) <= limit:
                            part = candidate
                        else:
                            if part:
                                chunks.append(part)
                            # Если слово само длиннее лимита — режем жестко
                            while len(word) > limit:
                                chunks.append(word[:limit])
                                word = word[limit:]
                            part = word
                    if part:
                        current = part
                continue

            # Есть текущий буфер
            if len(current) + 1 + len(line) <= limit:
                current += ("\n" + line)
            else:
                chunks.append(current)
                current = ""
                if len(line) <= limit:
                    current = line
                else:
                    part = ""
                    for word in line.split(" "):
                        candidate = (part + " " + word).strip() if part else word
                        if len(candidate) <= limit:
                            part = candidate
                        else:
                            if part:
                                chunks.append(part)
                            while len(word) > limit:
                                chunks.append(word[:limit])
                                word = word[limit:]
                            part = word
                    if part:
                        current = part

    if current:
        chunks.append(current)
    return chunks


async def send_markdown_safe(chat_id: int, text: str) -> None:
    """
    Надежная отправка длинных и/или Markdown-ответов:
    1) Разбивает на части <= 4096
    2) Пытается отправить с parse_mode="Markdown"
    3) При ошибке парсинга — повторяет как plain text
    """
    for chunk in split_text_for_telegram(text):
        try:
            await bot.send_message(chat_id, chunk, parse_mode="Markdown")
        except TelegramBadRequest as e:
            if "can't parse entities" in str(e).lower() or "bad request" in str(e).lower():
                await bot.send_message(chat_id, chunk)
            else:
                raise


async def setup_bot_commands(bot: Bot) -> None:
    """
    Регистрирует команды бота, чтобы они отображались в меню Telegram.
    """
    commands = [
        BotCommand(command="start", description="🚀 Начать работу"),
        BotCommand(command="remind", description="⏰ Добавить напоминание"),
        BotCommand(command="reminders", description="📋 Список напоминаний"),
        BotCommand(command="delremind", description="🗑️ Удалить напоминание"),
        BotCommand(command="snooze", description="😴 Отложить напоминание"),
        BotCommand(command="help", description="🆘 Помощь и возможности"),
        BotCommand(command="clear", description="🧹 Очистить историю диалога"),
    ]
    await bot.set_my_commands(commands)


# ========= Мастер добавления напоминаний (inline) =========
WIZARD_TIMEOUT_SEC = 600
wizard_states: dict[int, dict] = {}


def _wizard_init(user_id: int) -> dict:
    st = {
        "updated_at": time.time(),
        "type": None,
        "date_once": None,
        "time_hhmm": None,
        "weekdays": [],
        "weekday_only": False,
        "weekend_only": False,
        "period_minutes": None,
        "window_start": None,
        "window_end": None,
        "text": None,
        "awaiting": None,
    }
    wizard_states[user_id] = st
    return st


def _wizard_get(user_id: int) -> Optional[dict]:
    st = wizard_states.get(user_id)
    if not st:
        return None
    if time.time() - st.get("updated_at", 0) > WIZARD_TIMEOUT_SEC:
        wizard_states.pop(user_id, None)
        return None
    return st


def _wizard_touch(st: dict) -> None:
    st["updated_at"] = time.time()


def _kb_remind_type() -> InlineKeyboardMarkup:
    rows = [
        [InlineKeyboardButton(text="Ежедневно", callback_data="r:type:daily"), InlineKeyboardButton(text="Разово (дата)", callback_data="r:type:once")],
        [InlineKeyboardButton(text="Будни", callback_data="r:type:wk"), InlineKeyboardButton(text="Выходные", callback_data="r:type:we")],
        [InlineKeyboardButton(text="Дни недели", callback_data="r:type:days")],
        [InlineKeyboardButton(text="Периодически", callback_data="r:type:periodic")],
    ]
    return InlineKeyboardMarkup(inline_keyboard=rows)


def _kb_hours() -> InlineKeyboardMarkup:
    hours = [f"{h:02d}" for h in range(6, 24)]
    rows = []
    row = []
    for i, h in enumerate(hours, 1):
        row.append(InlineKeyboardButton(text=h, callback_data=f"r:timeh:{h}"))
        if i % 6 == 0:
            rows.append(row)
            row = []
    if row:
        rows.append(row)
    rows.append([InlineKeyboardButton(text="Назад", callback_data="r:back:type")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def _kb_minutes() -> InlineKeyboardMarkup:
    mins = ["00", "15", "30", "45"]
    rows = [[InlineKeyboardButton(text=m, callback_data=f"r:timem:{m}") for m in mins]]
    rows.append([
        InlineKeyboardButton(text="+15", callback_data="r:timem:+15"),
        InlineKeyboardButton(text="+30", callback_data="r:timem:+30"),
    ])
    rows.append([InlineKeyboardButton(text="Назад", callback_data="r:back:hours")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def _kb_date_once() -> InlineKeyboardMarkup:
    rows = [
        [InlineKeyboardButton(text="Сегодня", callback_data="r:date:today"), InlineKeyboardButton(text="Завтра", callback_data="r:date:tomorrow")],
        [InlineKeyboardButton(text="Ввести дату (YYYY-MM-DD)", callback_data="r:date:ask")],
        [InlineKeyboardButton(text="Назад", callback_data="r:back:type")],
    ]
    return InlineKeyboardMarkup(inline_keyboard=rows)


def _kb_days_toggle(st_days: list[int]) -> InlineKeyboardMarkup:
    names = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"]
    rows = []
    row = []
    for idx, name in enumerate(names):
        mark = "✓" if idx in st_days else ""
        row.append(InlineKeyboardButton(text=f"{name}{mark}", callback_data=f"r:day:{idx}"))
        if (idx + 1) % 4 == 0:
            rows.append(row)
            row = []
    if row:
        rows.append(row)
    rows.append([InlineKeyboardButton(text="Далее", callback_data="r:days:next")])
    rows.append([InlineKeyboardButton(text="Назад", callback_data="r:back:type")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def _kb_period_presets() -> InlineKeyboardMarkup:
    presets = [30, 45, 60, 90, 120]
    rows = []
    row = []
    for i, p in enumerate(presets, 1):
        row.append(InlineKeyboardButton(text=f"{p} мин", callback_data=f"r:per:{p}"))
        if i % 3 == 0:
            rows.append(row)
            row = []
    if row:
        rows.append(row)
    rows.append([InlineKeyboardButton(text="Свои минуты", callback_data="r:per:cust")])
    rows.append([InlineKeyboardButton(text="Назад", callback_data="r:back:type")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def _kb_window_presets() -> InlineKeyboardMarkup:
    rows = [
        [InlineKeyboardButton(text="09:00–21:00", callback_data="r:win:09:00-21:00"), InlineKeyboardButton(text="08:00–22:00", callback_data="r:win:08:00-22:00")],
        [InlineKeyboardButton(text="24/7", callback_data="r:win:24:7"), InlineKeyboardButton(text="Своё окно", callback_data="r:win:cust")],
        [InlineKeyboardButton(text="Пропустить", callback_data="r:win:none")],
        [InlineKeyboardButton(text="Назад", callback_data="r:back:per")],
    ]
    return InlineKeyboardMarkup(inline_keyboard=rows)


def _kb_text_presets() -> InlineKeyboardMarkup:
    rows = [
        [InlineKeyboardButton(text="Вода", callback_data="r:text:water"), InlineKeyboardButton(text="Обед", callback_data="r:text:lunch")],
        [InlineKeyboardButton(text="Перекус", callback_data="r:text:snack"), InlineKeyboardButton(text="Сон", callback_data="r:text:sleep")],
        [InlineKeyboardButton(text="Свой текст", callback_data="r:text:custom")],
        [InlineKeyboardButton(text="Назад", callback_data="r:back:time")],
    ]
    return InlineKeyboardMarkup(inline_keyboard=rows)


def _kb_confirm() -> InlineKeyboardMarkup:
    rows = [[InlineKeyboardButton(text="Создать", callback_data="r:confirm:create")],
            [InlineKeyboardButton(text="Назад", callback_data="r:back:text")]]
    return InlineKeyboardMarkup(inline_keyboard=rows)


async def start_remind_wizard(message: Message) -> None:
    user_id = message.from_user.id
    _wizard_init(user_id)
    await message.answer("Выбери тип напоминания:", reply_markup=_kb_remind_type())


def is_user_allowed(user_id: int) -> bool:
    """
    Проверяет, разрешен ли пользователю доступ к боту.
    
    Args:
        user_id: ID пользователя в Telegram
        
    Returns:
        True если пользователь разрешен, False если нет
    """
    return user_id in ALLOWED_USERS


@dp.message(Command("start"))
async def cmd_start(message: Message) -> None:
    """
    Обработчик команды /start.
    Отправляет приветственное сообщение новому пользователю.
    
    Args:
        message: Объект сообщения от пользователя
    """
    user_id = message.from_user.id
    user_name = message.from_user.first_name
    
    # Очищаем историю, если пользователь заново запускает бота
    await storage.clear_history(user_id)
    
    # Формируем персонализированное приветствие
    welcome_text = f"Привет, {user_name}! 👋\n\n{WELCOME_MESSAGE}"
    
    await send_markdown_safe(message.chat.id, welcome_text)
    
    # Добавляем первое сообщение в историю
    await storage.add_message(user_id, "assistant", welcome_text)
    # Сохраняем chat_id для напоминаний
    await storage.set_chat_id(user_id, message.chat.id)


@dp.message(Command("help"))
async def cmd_help(message: Message) -> None:
    """
    Обработчик команды /help.
    Отправляет информацию о возможностях бота.
    
    Args:
        message: Объект сообщения от пользователя
    """
    await send_markdown_safe(message.chat.id, HELP_MESSAGE)
    await storage.add_message(message.from_user.id, "assistant", HELP_MESSAGE)
    await storage.set_chat_id(message.from_user.id, message.chat.id)


@dp.message(Command("clear"))
async def cmd_clear(message: Message) -> None:
    """
    Обработчик команды /clear.
    Очищает историю сообщений пользователя.
    
    Args:
        message: Объект сообщения от пользователя
    """
    user_id = message.from_user.id
    await storage.clear_history(user_id)
    await message.answer("История диалога очищена. Давай начнем заново!")
    await storage.set_chat_id(user_id, message.chat.id)


@dp.message(Command("remind"))
async def cmd_remind(message: Message) -> None:
    """
    Добавляет напоминание. Поддерживаемые форматы:
    - /remind HH:MM текст — каждый день
    - /remind YYYY-MM-DD HH:MM текст — одноразовое на дату
    - /remind будни HH:MM текст — по будням (пн-пт)
    - /remind выходные HH:MM текст — по выходным (сб-вс)
    - /remind пн,ср,пт HH:MM текст — по перечисленным дням (пн,вт,ср,чт,пт,сб,вс)
    - /remind каждые Nмин HH:MM-HH:MM текст
    """
    user_id = message.from_user.id
    await storage.set_chat_id(user_id, message.chat.id)

    text_cmd = message.text.strip()
    try:
        # Убираем саму команду
        body = text_cmd.split(maxsplit=1)[1]
    except Exception:
        await start_remind_wizard(message)
        return

    # Наборы для распознавания дней недели
    days_map = {
        "пн": 0, "вт": 1, "ср": 2, "чт": 3, "пт": 4, "сб": 5, "вс": 6,
    }

    date_once = None
    weekdays = None
    weekday_only = False
    weekend_only = False
    period_minutes = None
    window_start = None
    window_end = None

    # Попытки распознавания форматов
    # 1) одноразовое: YYYY-MM-DD HH:MM текст
    m = re.match(r"^(\d{4}-\d{2}-\d{2})\s+(\d{2}:\d{2})\s+(.+)$", body)
    if m:
        date_once, time_str, text = m.group(1), m.group(2), m.group(3)
    else:
        # 2) будни/выходные: (будни|выходные) HH:MM текст
        m2 = re.match(r"^(будни|выходные)\s+(\d{2}:\d{2})\s+(.+)$", body, re.IGNORECASE)
        if m2:
            kind, time_str, text = m2.group(1).lower(), m2.group(2), m2.group(3)
            if kind == "будни":
                weekday_only = True
            else:
                weekend_only = True
        else:
            # 3) список дней: пн,ср,пт HH:MM текст
            m3 = re.match(r"^([а-яё,\s]+)\s+(\d{2}:\d{2})\s+(.+)$", body, re.IGNORECASE)
            parsed_days = None
            if m3:
                days_part = m3.group(1)
                time_str = m3.group(2)
                text = m3.group(3)
                try:
                    parsed_days = [
                        days_map[d.strip().lower()]
                        for d in days_part.split(',') if d.strip().lower() in days_map
                    ]
                except Exception:
                    parsed_days = None
                if parsed_days:
                    weekdays = parsed_days
                else:
                    # если парс не удался, считаем как обычный формат
                    time_str = None
            else:
                time_str = None

            # 3.5) периодические: каждые <N>мин [HH:MM-HH:MM] текст
            if time_str is None:
                mper = re.match(r"^каждые\s+(\d+)\s*мин(?:\s+(\d{2}:\d{2})-(\d{2}:\d{2}))?\s+(.+)$", body, re.IGNORECASE)
                if mper:
                    period_minutes = int(mper.group(1))
                    window_start = mper.group(2)
                    window_end = mper.group(3)
                    text = mper.group(4)
                    # для совместимости требуем time_str, поставим начало окна либо 00:00
                    time_str = window_start if window_start else "00:00"

            if time_str is None:
                # 4) базовый: HH:MM текст
                m4 = re.match(r"^(\d{2}:\d{2})\s+(.+)$", body)
                if not m4:
                    await message.answer(
                        "Неверный формат. Примеры:\n"
                        "- /remind 13:00 обед\n"
                        "- /remind 2025-08-13 09:30 важный созвон\n"
                        "- /remind будни 08:00 пробежка\n"
                        "- /remind выходные 10:30 созвон с родителями\n"
                        "- /remind пн,ср,пт 19:00 спортзал\n"
                        "- /remind каждые 60мин 09:00-21:00 вода"
                    )
                    return
                time_str, text = m4.group(1), m4.group(2)

    try:
        reminder = await storage.add_reminder(
            user_id,
            time_str,
            text.strip(),
            date_once=date_once,
            weekdays=weekdays,
            weekday_only=weekday_only,
            weekend_only=weekend_only,
            period_minutes=period_minutes,
            window_start_hhmm=window_start,
            window_end_hhmm=window_end,
        )
        await message.answer(
            f"Напоминание добавлено: {reminder['time']} — {reminder['text']}"
        )
    except ValueError as e:
        await message.answer(str(e))


@dp.message(Command("reminders"))
async def cmd_reminders(message: Message) -> None:
    """
    Показывает список напоминаний пользователя.
    """
    user_id = message.from_user.id
    await storage.set_chat_id(user_id, message.chat.id)
    reminders = await storage.list_reminders(user_id)
    if not reminders:
        await message.answer(
            "У вас пока нет напоминаний.\n"
            "Пример: /remind 13:00 обед\n"
            "Также можно: /remind 08:30 зарядка"
        )
        return
    # Формируем подробный список и инлайн-кнопки
    lines = []
    keyboard_rows = []
    for r in reminders:
        extra_parts = []
        if r.get('date_once'):
            extra_parts.append(f"дата: {r['date_once']}")
        if r.get('weekday_only'):
            extra_parts.append("будни")
        if r.get('weekend_only'):
            extra_parts.append("выходные")
        if r.get('weekdays'):
            # строки вида "0,2,4" -> пн,ср,пт
            idx_to_ru = {0:"пн",1:"вт",2:"ср",3:"чт",4:"пт",5:"сб",6:"вс"}
            try:
                items = [idx_to_ru.get(int(x), str(x)) for x in r['weekdays'].split(',') if x]
                if items:
                    extra_parts.append("дни: " + ",".join(items))
            except Exception:
                pass
        if r.get('snooze_until'):
            extra_parts.append(f"отложено до {r['snooze_until']}")
        extras = f" ({'; '.join(extra_parts)})" if extra_parts else ""
        lines.append(f"{r['time']} — {r['text']}{extras}")
        keyboard_rows.append([
            InlineKeyboardButton(text=f"Удалить {r['time']} — {r['text']}", callback_data=f"delremind:{r['id']}")
        ])

    kb = InlineKeyboardMarkup(inline_keyboard=keyboard_rows)
    await message.answer("Ваши напоминания:\n" + "\n".join(lines), reply_markup=kb)


@dp.message(Command("delremind"))
async def cmd_delremind(message: Message) -> None:
    """
    Удаляет напоминание по ID: /delremind ID
    """
    user_id = message.from_user.id
    parts = message.text.split(maxsplit=1)
    if len(parts) < 2:
        # Если ID не указан — покажем список с кнопками для удаления
        reminders = await storage.list_reminders(user_id)
        if not reminders:
            await message.answer("Нет напоминаний для удаления. Сначала добавьте: /remind 13:00 обед")
            return
        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text=f"Удалить {r['time']} — {r['text']}", callback_data=f"delremind:{r['id']}")]
            for r in reminders
        ])
        await message.answer("Выберите напоминание для удаления:", reply_markup=keyboard)
        return
    try:
        identifier = int(parts[1])
    except ValueError:
        await message.answer("ID должен быть числом")
        return
    ok = await storage.delete_reminder(user_id, identifier)
    if ok:
        await message.answer("Напоминание удалено")
    else:
        await message.answer("Не нашёл напоминание с таким ID")


@dp.callback_query(F.data.startswith("delremind:"))
async def cb_delremind(callback: CallbackQuery) -> None:
    """
    Обработчик нажатия на кнопку удаления напоминания.
    """
    user_id = callback.from_user.id if callback.from_user else None
    if user_id is None or user_id not in ALLOWED_USERS:
        await callback.answer("🚫 Нет доступа", show_alert=True)
        return

    try:
        identifier_str = callback.data.split(":", 1)[1]
        identifier = int(identifier_str)
    except Exception:
        await callback.answer("Некорректный идентификатор", show_alert=True)
        return

    ok = await storage.delete_reminder(user_id, identifier)
    if ok:
        await callback.answer("Напоминание удалено")
        await callback.message.answer("Напоминание удалено")
    else:
        await callback.answer("Не найдено", show_alert=True)


@dp.callback_query(F.data.startswith("snooze:"))
async def cb_snooze(callback: CallbackQuery) -> None:
    """
    Обработчик "Отложить на N минут" из инлайн-кнопки.
    Формат callback_data: snooze:<id>:<minutes>
    """
    user_id = callback.from_user.id if callback.from_user else None
    if user_id is None or user_id not in ALLOWED_USERS:
        await callback.answer("🚫 Нет доступа", show_alert=True)
        return

    try:
        _, rid_str, minutes_str = callback.data.split(":", 2)
        rid = int(rid_str)
        minutes = int(minutes_str)
    except Exception:
        await callback.answer("Некорректные параметры", show_alert=True)
        return

    now_local = datetime.now(LOCAL_TZ)
    snooze_until = now_local + timedelta(minutes=minutes)
    snooze_iso = snooze_until.strftime("%Y-%m-%d %H:%M")
    await storage.set_reminder_snooze(user_id, rid, snooze_iso)
    await callback.answer(f"Отложено на {minutes} мин")
    try:
        await callback.message.edit_reply_markup(reply_markup=None)
    except Exception:
        pass


# ====== Коллбэки мастера напоминаний ======

@dp.callback_query(F.data.startswith("r:type:"))
async def cb_r_type(callback: CallbackQuery) -> None:
    user_id = callback.from_user.id
    st = _wizard_get(user_id) or _wizard_init(user_id)
    _wizard_touch(st)
    typ = callback.data.split(":", 2)[2]
    st["type"] = typ
    if typ == "daily":
        await callback.message.edit_text("Выбери час:", reply_markup=_kb_hours())
    elif typ == "once":
        await callback.message.edit_text("Выбери дату:", reply_markup=_kb_date_once())
    elif typ in ("wk", "we"):
        st["weekday_only"] = (typ == "wk")
        st["weekend_only"] = (typ == "we")
        await callback.message.edit_text("Выбери час:", reply_markup=_kb_hours())
    elif typ == "days":
        await callback.message.edit_text("Выбери дни недели:", reply_markup=_kb_days_toggle(st["weekdays"]))
    elif typ == "periodic":
        await callback.message.edit_text("Период (минуты):", reply_markup=_kb_period_presets())
    await callback.answer()


@dp.callback_query(F.data.startswith("r:timeh:"))
async def cb_r_time_hour(callback: CallbackQuery) -> None:
    user_id = callback.from_user.id
    st = _wizard_get(user_id) or _wizard_init(user_id)
    _wizard_touch(st)
    hour = callback.data.split(":", 2)[2]
    st["time_hhmm"] = f"{hour}:00"
    await callback.message.edit_text("Выбери минуты:", reply_markup=_kb_minutes())
    await callback.answer()


@dp.callback_query(F.data.startswith("r:timem:"))
async def cb_r_time_min(callback: CallbackQuery) -> None:
    user_id = callback.from_user.id
    st = _wizard_get(user_id) or _wizard_init(user_id)
    _wizard_touch(st)
    val = callback.data.split(":", 2)[2]
    hh, mm = (st.get("time_hhmm") or "00:00").split(":")
    if val.startswith("+"):
        inc = 15 if "+15" in val else 30
        total = int(hh) * 60 + int(mm) + inc
        nh = (total // 60) % 24
        nm = total % 60
        st["time_hhmm"] = f"{nh:02d}:{nm:02d}"
    else:
        st["time_hhmm"] = f"{hh}:{val}"
    await callback.message.edit_text(f"Время: {st['time_hhmm']}\nВыбери текст или пресет:", reply_markup=_kb_text_presets())
    await callback.answer()


@dp.callback_query(F.data.startswith("r:date:"))
async def cb_r_date(callback: CallbackQuery) -> None:
    user_id = callback.from_user.id
    st = _wizard_get(user_id) or _wizard_init(user_id)
    _wizard_touch(st)
    what = callback.data.split(":", 2)[2]
    now = datetime.now(LOCAL_TZ)
    if what == "today":
        st["date_once"] = now.strftime("%Y-%m-%d")
        await callback.message.edit_text("Выбери час:", reply_markup=_kb_hours())
    elif what == "tomorrow":
        st["date_once"] = (now + timedelta(days=1)).strftime("%Y-%m-%d")
        await callback.message.edit_text("Выбери час:", reply_markup=_kb_hours())
    elif what == "ask":
        st["awaiting"] = "date_once"
        await callback.message.edit_text("Введи дату формата YYYY-MM-DD")
    await callback.answer()


@dp.callback_query(F.data.startswith("r:day:"))
async def cb_r_day_toggle(callback: CallbackQuery) -> None:
    user_id = callback.from_user.id
    st = _wizard_get(user_id) or _wizard_init(user_id)
    _wizard_touch(st)
    idx = int(callback.data.split(":", 2)[2])
    days = set(st.get("weekdays") or [])
    if idx in days:
        days.remove(idx)
    else:
        days.add(idx)
    st["weekdays"] = sorted(days)
    try:
        await callback.message.edit_reply_markup(reply_markup=_kb_days_toggle(st["weekdays"]))
    except Exception:
        await callback.message.edit_text("Выбери дни недели:", reply_markup=_kb_days_toggle(st["weekdays"]))
    await callback.answer()


@dp.callback_query(F.data == "r:days:next")
async def cb_r_days_next(callback: CallbackQuery) -> None:
    user_id = callback.from_user.id
    st = _wizard_get(user_id) or _wizard_init(user_id)
    _wizard_touch(st)
    await callback.message.edit_text("Выбери час:", reply_markup=_kb_hours())
    await callback.answer()


@dp.callback_query(F.data.startswith("r:per:"))
async def cb_r_period(callback: CallbackQuery) -> None:
    user_id = callback.from_user.id
    st = _wizard_get(user_id) or _wizard_init(user_id)
    _wizard_touch(st)
    what = callback.data.split(":", 2)[2]
    if what == "cust":
        st["awaiting"] = "period"
        await callback.message.edit_text("Введи период в минутах, например 60")
    else:
        st["period_minutes"] = int(what)
        await callback.message.edit_text("Окно (необязательно):", reply_markup=_kb_window_presets())
    await callback.answer()


@dp.callback_query(F.data.startswith("r:win:"))
async def cb_r_window(callback: CallbackQuery) -> None:
    user_id = callback.from_user.id
    st = _wizard_get(user_id) or _wizard_init(user_id)
    _wizard_touch(st)
    data = callback.data.split(":", 2)[2]
    if data == "none":
        st["window_start"], st["window_end"] = None, None
    elif data == "cust":
        st["awaiting"] = "window"
        await callback.message.edit_text("Введи окно формата HH:MM-HH:MM, например 09:00-21:00")
        await callback.answer()
        return
    elif data == "24:7":
        st["window_start"], st["window_end"] = None, None
    else:
        try:
            ws, we = data.split("-")
            st["window_start"], st["window_end"] = ws, we
        except Exception:
            st["window_start"], st["window_end"] = None, None
    await callback.message.edit_text("Текст напоминания:", reply_markup=_kb_text_presets())
    await callback.answer()


@dp.callback_query(F.data.startswith("r:text:"))
async def cb_r_text(callback: CallbackQuery) -> None:
    user_id = callback.from_user.id
    st = _wizard_get(user_id) or _wizard_init(user_id)
    _wizard_touch(st)
    what = callback.data.split(":", 2)[2]
    presets = {"water": "вода", "lunch": "обед", "snack": "перекус", "sleep": "сон"}
    if what == "custom":
        st["awaiting"] = "text"
        await callback.message.edit_text("Введи текст напоминания сообщением")
    else:
        st["text"] = presets.get(what, what)
        await callback.message.edit_text(_wizard_summary(st), reply_markup=_kb_confirm())
    await callback.answer()


def _wizard_summary(st: dict) -> str:
    parts = ["Подтвердите напоминание:"]
    t = st.get("type")
    if t == "periodic":
        per = st.get("period_minutes")
        win = (st.get("window_start"), st.get("window_end"))
        parts.append(f"- Тип: периодически каждые {per} мин")
        if any(win):
            parts.append(f"- Окно: {win[0] or '00:00'}-{win[1] or '24:00'}")
    else:
        parts.append(f"- Время: {st.get('time_hhmm')}")
    if t == "once":
        parts.append(f"- Дата: {st.get('date_once')}")
    if t == "wk":
        parts.append("- Дни: будни")
    if t == "we":
        parts.append("- Дни: выходные")
    if t == "days" and st.get("weekdays"):
        idx_to_ru = {0:"пн",1:"вт",2:"ср",3:"чт",4:"пт",5:"сб",6:"вс"}
        parts.append("- Дни: " + ",".join(idx_to_ru[i] for i in st["weekdays"]))
    parts.append(f"- Текст: {st.get('text')}")
    return "\n".join(parts)


@dp.callback_query(F.data == "r:back:type")
async def cb_r_back_type(callback: CallbackQuery) -> None:
    await callback.message.edit_text("Выбери тип напоминания:", reply_markup=_kb_remind_type())
    await callback.answer()


@dp.callback_query(F.data == "r:back:hours")
async def cb_r_back_hours(callback: CallbackQuery) -> None:
    await callback.message.edit_text("Выбери час:", reply_markup=_kb_hours())
    await callback.answer()


@dp.callback_query(F.data == "r:back:time")
async def cb_r_back_time(callback: CallbackQuery) -> None:
    await callback.message.edit_text("Выбери минуты:", reply_markup=_kb_minutes())
    await callback.answer()


@dp.callback_query(F.data == "r:back:per")
async def cb_r_back_per(callback: CallbackQuery) -> None:
    await callback.message.edit_text("Период (минуты):", reply_markup=_kb_period_presets())
    await callback.answer()


@dp.callback_query(F.data == "r:back:text")
async def cb_r_back_text(callback: CallbackQuery) -> None:
    await callback.message.edit_text("Текст напоминания:", reply_markup=_kb_text_presets())
    await callback.answer()


@dp.callback_query(F.data == "r:confirm:create")
async def cb_r_confirm_create(callback: CallbackQuery) -> None:
    user_id = callback.from_user.id
    st = _wizard_get(user_id)
    if not st:
        await callback.answer("Мастер устарел, начните заново: /remind", show_alert=True)
        return
    _wizard_touch(st)
    typ = st.get("type")
    time_hhmm = st.get("time_hhmm") or "00:00"
    text = st.get("text") or "напоминание"
    date_once = st.get("date_once")
    weekdays = st.get("weekdays") or None
    weekday_only = bool(st.get("weekday_only"))
    weekend_only = bool(st.get("weekend_only"))
    period_minutes = st.get("period_minutes")
    window_start = st.get("window_start")
    window_end = st.get("window_end")

    try:
        reminder = await storage.add_reminder(
            user_id,
            time_hhmm,
            text.strip(),
            date_once=date_once,
            weekdays=weekdays,
            weekday_only=weekday_only,
            weekend_only=weekend_only,
            period_minutes=period_minutes,
            window_start_hhmm=window_start,
            window_end_hhmm=window_end,
        )
        wizard_states.pop(user_id, None)
        await callback.message.edit_text(
            f"Напоминание добавлено: {reminder['time']} — {reminder['text']}"
        )
    except Exception as e:
        await callback.answer(str(e), show_alert=True)


@dp.message(Command("snooze"))
async def cmd_snooze(message: Message) -> None:
    """
    Отложить напоминание: /snooze ID [минуты], по умолчанию 10 минут
    """
    user_id = message.from_user.id
    parts = message.text.split()
    if len(parts) < 2:
        await message.answer("Формат: /snooze ID [минуты]. Пример: /snooze 5 10")
        return
    try:
        rid = int(parts[1])
        minutes = int(parts[2]) if len(parts) >= 3 else 10
    except Exception:
        await message.answer("Некорректные параметры. Пример: /snooze 5 10")
        return
    now_local = datetime.now(LOCAL_TZ)
    snooze_until = now_local + timedelta(minutes=minutes)
    snooze_iso = snooze_until.strftime("%Y-%m-%d %H:%M")
    storage.set_reminder_snooze(user_id, rid, snooze_iso)
    await message.answer(f"Напоминание [{rid}] отложено на {minutes} минут")


@dp.message(F.text)
async def handle_message(message: Message) -> None:
    """
    Обработчик текстовых сообщений от пользователя.
    Отправляет запрос к OpenAI и возвращает ответ.
    
    Args:
        message: Объект сообщения от пользователя
    """
    user_id = message.from_user.id
    # Перехват ввода для мастера (дата/период/окно/текст)
    st = _wizard_get(user_id)
    if st and st.get("awaiting"):
        kind = st["awaiting"]
        st["awaiting"] = None
        _wizard_touch(st)
        if kind == "date_once":
            if re.match(r"^\d{4}-\d{2}-\d{2}$", message.text.strip()):
                st["date_once"] = message.text.strip()
                await message.answer("Выбери час:", reply_markup=_kb_hours())
                return
            else:
                await message.answer("Формат даты: YYYY-MM-DD. Попробуй ещё раз.")
                st["awaiting"] = "date_once"
                return
        if kind == "period":
            try:
                st["period_minutes"] = int(message.text.strip())
                await message.answer("Окно (необязательно):", reply_markup=_kb_window_presets())
                return
            except Exception:
                await message.answer("Введи число минут, например 60")
                st["awaiting"] = "period"
                return
        if kind == "window":
            m = re.match(r"^(\d{2}:\d{2})-(\d{2}:\d{2})$", message.text.strip())
            if m:
                st["window_start"], st["window_end"] = m.group(1), m.group(2)
                await message.answer("Текст напоминания:", reply_markup=_kb_text_presets())
                return
            else:
                await message.answer("Формат окна: HH:MM-HH:MM. Попробуй ещё раз.")
                st["awaiting"] = "window"
                return
        if kind == "text":
            st["text"] = message.text.strip()
            await message.answer(_wizard_summary(st), reply_markup=_kb_confirm())
            return
    user_message = message.text
    
    # Используем ID чата для отправки ответа
    chat_id = message.chat.id
    
    # Сохраняем сообщение пользователя
    await storage.add_message(user_id, "user", user_message)
    await storage.set_chat_id(user_id, chat_id)
    
    # Индикатор печати
    await bot.send_chat_action(chat_id=chat_id, action="typing")
    
    # Получаем историю сообщений пользователя
    message_history = await storage.get_message_history(user_id)
    
    try:
        # Получаем ответ от Vibe Checker
        response_text = await get_vibe_checker_response(message_history, user_id=user_id)
        
        # Отправляем ответ без добавления упоминания пользователя
        await send_markdown_safe(chat_id, response_text)
        
        # Сохраняем ответ в историю
        await storage.add_message(user_id, "assistant", response_text)
        
    except Exception as e:
        logger.error(f"Ошибка при обработке сообщения: {str(e)}")
        await message.answer("Произошла ошибка при обработке запроса. Пожалуйста, попробуйте еще раз.")


@dp.message(F.voice)
async def handle_voice_message(message: Message) -> None:
    """
    Обработчик голосовых сообщений от пользователя.
    Преобразует голосовое сообщение в текст и обрабатывает его.
    
    Args:
        message: Объект сообщения от пользователя
    """
    user_id = message.from_user.id
    chat_id = message.chat.id
    
    # Отправляем промежуточное сообщение
    await message.answer("Обрабатываю голосовое сообщение...")
    
    try:
        # Получаем голосовое сообщение
        voice = message.voice
        file = await bot.get_file(voice.file_id)
        
        # Скачиваем голосовое сообщение в память
        voice_io = io.BytesIO()
        await bot.download(file, destination=voice_io)
        voice_io.seek(0)  # Перемещаем указатель в начало файла
        voice_data = voice_io.read()
        
        # Транскрибируем голосовое сообщение в текст
        voice_text = await transcribe_voice(voice_data, f"{voice.file_id}.ogg")
        
        # Отправляем подтверждение распознавания
        await message.answer(f"Ваше сообщение: {voice_text}")
        
        # Сохраняем текст сообщения в историю пользователя
        await storage.add_message(user_id, "user", voice_text)
        await storage.set_chat_id(user_id, chat_id)
        
        # Индикатор печати
        await bot.send_chat_action(chat_id=chat_id, action="typing")
        
        # Получаем историю сообщений пользователя
        message_history = await storage.get_message_history(user_id)
        
        # Получаем ответ от Vibe Checker
        response_text = await get_vibe_checker_response(message_history, user_id=user_id)
        
        # Отправляем ответ пользователю
        await send_markdown_safe(chat_id, response_text)
        
        # Сохраняем ответ в историю
        await storage.add_message(user_id, "assistant", response_text)
        
    except Exception as e:
        logger.error(f"Ошибка при обработке голосового сообщения: {str(e)}")
        await message.answer("Произошла ошибка при обработке голосового сообщения. Пожалуйста, попробуйте еще раз.")


async def main() -> None:
    """
    Основная функция для запуска бота.
    """
    logger.info("Запуск бота...")
    
    # Пропускаем накопившиеся обновления
    await bot.delete_webhook(drop_pending_updates=True)

    # Регистрируем команды бота для меню Telegram
    await setup_bot_commands(bot)

    # Запускаем фоновую задачу отправки напоминаний
    async def reminders_loop():
        while True:
            try:
                now_local = datetime.now(LOCAL_TZ)
                hhmm = now_local.strftime("%H:%M")
                today = now_local.strftime("%Y-%m-%d")
                weekday_idx = int(now_local.strftime("%w")) - 1  # 0=пн .. 6=вс
                if weekday_idx < 0:
                    weekday_idx = 6
                now_iso = now_local.strftime("%Y-%m-%d %H:%M")

                due = await storage.get_due_reminders(hhmm, today, weekday_idx, now_iso)
                for user_id_int, reminder, is_snooze in due:
                    chat_id_saved = await storage.get_chat_id(user_id_int)
                    if chat_id_saved:
                        try:
                            # Инлайн-кнопка для отложить на 10 минут
                            kb = InlineKeyboardMarkup(
                                inline_keyboard=[
                                    [InlineKeyboardButton(text="Отложить на 10 мин", callback_data=f"snooze:{reminder['id']}:10")]
                                ]
                            )
                            await bot.send_message(chat_id_saved, f"⏰ Напоминание: {reminder['text']}", reply_markup=kb)
                            if is_snooze:
                                await storage.clear_snooze(user_id_int, int(reminder['id']))
                            else:
                                await storage.mark_reminder_sent(user_id_int, int(reminder['id']), today)
                            # Если напоминание периодическое — двигаем next_fire_at
                            if reminder.get('periodic') is True:
                                await storage.bump_periodic_next_fire(user_id_int, int(reminder['id']), now_iso)
                        except Exception as send_err:
                            logger.error(f"Не удалось отправить напоминание {reminder}: {send_err}")
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                logger.info("Цикл напоминаний отменён, выходим...")
                break
            except Exception as loop_err:
                logger.error(f"Ошибка в цикле напоминаний: {loop_err}")
                await asyncio.sleep(30)

    reminders_task = asyncio.create_task(reminders_loop())

    try:
        # Старт поллинга
        await dp.start_polling(bot)
    finally:
        # Корректно останавливаем фоновую задачу напоминаний
        reminders_task.cancel()
        with contextlib.suppress(Exception):
            await asyncio.gather(reminders_task)


if __name__ == "__main__":
    try:
        # Запускаем бота
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Бот остановлен!")
    except Exception as e:
        logger.error(f"Критическая ошибка: {str(e)}") 