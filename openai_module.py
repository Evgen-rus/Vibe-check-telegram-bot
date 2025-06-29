"""
Модуль для взаимодействия с OpenAI API.
"""

import openai
from typing import List, Dict, Any, Optional
import asyncio
from datetime import datetime
import pytz
from config import OPENAI_API_KEY, OPENAI_MODEL, OPENAI_PARAMS, logger
from prompts import SYSTEM_PROMPT

# Настройка клиента OpenAI
client = openai.OpenAI(api_key=OPENAI_API_KEY)


def get_novosibirsk_time() -> str:
    """
    Получает текущее время в Новосибирске (UTC+7).
    
    Returns:
        Строка с текущим временем в формате для AI
    """
    # Создаем объект временной зоны Новосибирска (UTC+7)
    novosibirsk_tz = pytz.timezone('Asia/Novosibirsk')
    
    # Получаем текущее время в Новосибирске
    current_time = datetime.now(novosibirsk_tz)
    
    # Форматируем время для передачи AI
    formatted_time = current_time.strftime("Текущее время: %d.%m.%Y, %H:%M, %A, Новосибирск")
    
    # Переводим день недели на русский
    days_translation = {
        'Monday': 'понедельник',
        'Tuesday': 'вторник', 
        'Wednesday': 'среда',
        'Thursday': 'четверг',
        'Friday': 'пятница',
        'Saturday': 'суббота',
        'Sunday': 'воскресенье'
    }
    
    day_english = current_time.strftime("%A")
    day_russian = days_translation.get(day_english, day_english)
    
    return formatted_time.replace(day_english, day_russian)


async def generate_response(
    messages: List[Dict[str, str]], 
    system_prompt: Optional[str] = None
) -> str:
    """
    Генерирует ответ с использованием API OpenAI.
    
    Args:
        messages: Список сообщений в формате OpenAI
        system_prompt: Системный промпт (опционально)
        
    Returns:
        Строка с ответом модели
        
    Raises:
        Exception: Если произошла ошибка при запросе к API
    """
    try:
        # Добавляем системный промпт, если он предоставлен
        full_messages = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})
        full_messages.extend(messages)
        
        # Создаем запрос к API OpenAI
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model=OPENAI_MODEL,
            messages=full_messages,
            temperature=OPENAI_PARAMS.get("temperature", 0.8),
            max_tokens=OPENAI_PARAMS.get("max_tokens", 1000),
        )
        
        # Извлекаем ответ модели
        if response.choices and len(response.choices) > 0:
            return response.choices[0].message.content
        else:
            logger.error("Пустой ответ от OpenAI API")
            return "Извините, произошла ошибка при обработке запроса."
            
    except Exception as e:
        logger.error(f"Ошибка при запросе к OpenAI API: {str(e)}")
        return "Извините, произошла ошибка при обработке запроса. Пожалуйста, повторите попытку позже."


async def get_vibe_checker_response(user_messages: List[Dict[str, str]]) -> str:
    """
    Получает ответ от Vibe Checker с временным контекстом.
    
    Args:
        user_messages: История сообщений пользователя
        
    Returns:
        Строка с ответом Vibe Checker
    """
    # Получаем текущее время Новосибирска
    time_context = get_novosibirsk_time()
    
    # Добавляем временной контекст к системному промпту
    enhanced_system_prompt = f"{SYSTEM_PROMPT}\n\n{time_context}"
    
    return await generate_response(user_messages, enhanced_system_prompt) 