"""
Vibe Checker - персональный помощник по питанию.
Использует OpenAI API для понимания естественной речи и поддерживает голосовые сообщения.

Usage:
    python main.py
"""

import asyncio
import io
import contextlib
from typing import List
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
        BotCommand(command="start", description="Начать работу"),
        BotCommand(command="help", description="Помощь и возможности"),
        BotCommand(command="clear", description="Очистить историю диалога"),
        BotCommand(command="remind", description="Добавить напоминание"),
        BotCommand(command="reminders", description="Список напоминаний"),
        BotCommand(command="delremind", description="Удалить: /delremind ID"),
        BotCommand(command="snooze", description="Отложить: /snooze ID [мин]"),
    ]
    await bot.set_my_commands(commands)


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
    """
    user_id = message.from_user.id
    await storage.set_chat_id(user_id, message.chat.id)

    text_cmd = message.text.strip()
    try:
        # Убираем саму команду
        body = text_cmd.split(maxsplit=1)[1]
    except Exception:
        await message.answer(
            "Форматы:\n"
            "- /remind HH:MM текст\n"
            "- /remind YYYY-MM-DD HH:MM текст\n"
            "- /remind будни HH:MM текст\n"
            "- /remind выходные HH:MM текст\n"
            "- /remind пн,ср,пт HH:MM текст"
        )
        return

    # Наборы для распознавания дней недели
    days_map = {
        "пн": 0, "вт": 1, "ср": 2, "чт": 3, "пт": 4, "сб": 5, "вс": 6,
    }

    date_once = None
    weekdays = None
    weekday_only = False
    weekend_only = False

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
                        "- /remind пн,ср,пт 19:00 спортзал"
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
        )
        await message.answer(
            f"Напоминание добавлено: [{reminder['id']}] {reminder['time']} — {reminder['text']}"
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
    lines = [f"[{r['id']}] {r['time']} — {r['text']}" for r in reminders]
    await message.answer("Ваши напоминания:\n" + "\n".join(lines))


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
            [InlineKeyboardButton(text=f"Удалить [{r['id']}] {r['time']} — {r['text']}", callback_data=f"delremind:{r['id']}")]
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
    storage.set_reminder_snooze(user_id, rid, snooze_iso)
    await callback.answer(f"Отложено на {minutes} мин")
    try:
        await callback.message.edit_reply_markup(reply_markup=None)
    except Exception:
        pass


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
        response_text = await get_vibe_checker_response(message_history)
        
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
        response_text = await get_vibe_checker_response(message_history)
        
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