"""
Модуль для взаимодействия с OpenAI API.
"""

import openai
from typing import List, Dict, Any, Optional
import asyncio
from config import OPENAI_API_KEY, OPENAI_MODEL, OPENAI_PARAMS, logger
from prompts import SYSTEM_PROMPT

# Настройка клиента OpenAI
client = openai.OpenAI(api_key=OPENAI_API_KEY)


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


async def get_recipe_response(user_messages: List[Dict[str, str]]) -> str:
    """
    Получает ответ с рецептом от OpenAI.
    
    Args:
        user_messages: История сообщений пользователя
        
    Returns:
        Строка с ответом модели, содержащим рецепт
    """
    return await generate_response(user_messages, SYSTEM_PROMPT) 