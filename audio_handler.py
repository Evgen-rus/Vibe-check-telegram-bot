"""
Модуль для обработки голосовых сообщений и их преобразования в текст.
"""

import io
import openai
from openai import AsyncOpenAI
from config import OPENAI_API_KEY, TRANSCRIPTION_MODEL, logger

# Инициализация клиента OpenAI
client = AsyncOpenAI(api_key=OPENAI_API_KEY)

async def transcribe_voice(voice_data: bytes, file_name: str = "voice.ogg", language: str = "ru") -> str:
    """
    Асинхронная функция для транскрибации голосового сообщения в текст.
    
    Args:
        voice_data: Байтовые данные голосового сообщения
        file_name: Имя файла для отправки в API
        language: Язык голосового сообщения для лучшего распознавания
        
    Returns:
        str: Распознанный текст
        
    Raises:
        Exception: В случае ошибки при транскрибации
    """
    try:
        logger.info(f"Начинаю транскрибацию голосового сообщения, размер: {len(voice_data)} байт")
        
        # Отправляем запрос на транскрибацию
        transcript = await client.audio.transcriptions.create(
            model=TRANSCRIPTION_MODEL,
            file=(file_name, voice_data),
            language=language
        )
        
        # Получаем и логируем результат
        text = transcript.text
        logger.info(f"Голосовое сообщение успешно транскрибировано: {text[:50]}...")
        
        return text
        
    except Exception as e:
        # Если произошла ошибка с моделью, пробуем запасную модель
        if "invalid model ID" in str(e):
            logger.warning(f"Модель {TRANSCRIPTION_MODEL} недоступна, используем whisper-1")
            try:
                transcript = await client.audio.transcriptions.create(
                    model="whisper-1",  # Запасная модель
                    file=(file_name, voice_data),
                    language=language
                )
                text = transcript.text
                logger.info(f"Голосовое сообщение транскрибировано запасной моделью: {text[:50]}...")
                return text
                
            except Exception as inner_e:
                logger.error(f"Ошибка при использовании запасной модели: {inner_e}")
                raise
        else:
            logger.error(f"Ошибка при транскрибации: {e}")
            raise 