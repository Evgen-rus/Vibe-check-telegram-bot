"""
Vibe Checker - персональный помощник по делам и продуктивности.
Телеграм-бот для планирования задач, отслеживания выполнения и повышения продуктивности.

Использует OpenAI API для понимания естественной речи и поддерживает голосовые сообщения.

Usage:
    python main.py
"""

import asyncio
import logging
import io
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.types import Message
from aiogram import F

from config import TELEGRAM_BOT_TOKEN, logger
from openai_module import get_vibe_checker_response
from storage import storage
from prompts import WELCOME_MESSAGE, HELP_MESSAGE
from audio_handler import transcribe_voice

# Инициализация бота и диспетчера
bot = Bot(token=TELEGRAM_BOT_TOKEN)
dp = Dispatcher()


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
    storage.clear_history(user_id)
    
    # Формируем персонализированное приветствие
    welcome_text = f"Привет, {user_name}! 👋\n\n{WELCOME_MESSAGE}"
    
    await message.answer(welcome_text)
    
    # Добавляем первое сообщение в историю
    storage.add_message(user_id, "assistant", welcome_text)


@dp.message(Command("help"))
async def cmd_help(message: Message) -> None:
    """
    Обработчик команды /help.
    Отправляет информацию о возможностях бота.
    
    Args:
        message: Объект сообщения от пользователя
    """
    await message.answer(HELP_MESSAGE)
    storage.add_message(message.from_user.id, "assistant", HELP_MESSAGE)


@dp.message(Command("clear"))
async def cmd_clear(message: Message) -> None:
    """
    Обработчик команды /clear.
    Очищает историю сообщений пользователя.
    
    Args:
        message: Объект сообщения от пользователя
    """
    user_id = message.from_user.id
    storage.clear_history(user_id)
    await message.answer("История диалога очищена. Давай начнем заново!")


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
    storage.add_message(user_id, "user", user_message)
    
    # Индикатор печати
    await bot.send_chat_action(chat_id=chat_id, action="typing")
    
    # Получаем историю сообщений пользователя
    message_history = storage.get_message_history(user_id)
    
    try:
        # Получаем ответ от Vibe Checker
        response_text = await get_vibe_checker_response(message_history)
        
        # Отправляем ответ без добавления упоминания пользователя
        await message.answer(response_text, parse_mode="Markdown")
        
        # Сохраняем ответ в историю
        storage.add_message(user_id, "assistant", response_text)
        
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
        storage.add_message(user_id, "user", voice_text)
        
        # Индикатор печати
        await bot.send_chat_action(chat_id=chat_id, action="typing")
        
        # Получаем историю сообщений пользователя
        message_history = storage.get_message_history(user_id)
        
        # Получаем ответ от Vibe Checker
        response_text = await get_vibe_checker_response(message_history)
        
        # Отправляем ответ пользователю
        await message.answer(response_text, parse_mode="Markdown")
        
        # Сохраняем ответ в историю
        storage.add_message(user_id, "assistant", response_text)
        
    except Exception as e:
        logger.error(f"Ошибка при обработке голосового сообщения: {str(e)}")
        await message.answer("Произошла ошибка при обработке голосового сообщения. Пожалуйста, попробуйте еще раз.")


async def main() -> None:
    """
    Основная функция для запуска бота.
    """
    logger.info("Запуск бота...")
    
    # Пропускаем накопившиеся обновления и запускаем поллинг
    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot)


if __name__ == "__main__":
    try:
        # Запускаем бота
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Бот остановлен!")
    except Exception as e:
        logger.error(f"Критическая ошибка: {str(e)}") 