"""
Модуль для взаимодействия с OpenAI Responses API.
"""

from typing import List, Dict, Optional
import asyncio
from datetime import datetime
import pytz
from openai import OpenAI
from config import OPENAI_API_KEY, OPENAI_MODEL, OPENAI_PARAMS, logger
from prompts import SYSTEM_PROMPT

# Настройка клиента OpenAI (Responses API)
client = OpenAI(api_key=OPENAI_API_KEY)


def get_novosibirsk_time() -> str:
    """
    Получает текущее время в Новосибирске (UTC+7).

    Returns:
        Строка с текущим временем в формате для AI
    """
    novosibirsk_tz = pytz.timezone("Asia/Novosibirsk")
    current_time = datetime.now(novosibirsk_tz)

    formatted_time = current_time.strftime(
        "Текущее время: %d.%m.%Y, %H:%M, %A, Новосибирск"
    )

    days_translation = {
        "Monday": "понедельник",
        "Tuesday": "вторник",
        "Wednesday": "среда",
        "Thursday": "четверг",
        "Friday": "пятница",
        "Saturday": "суббота",
        "Sunday": "воскресенье",
    }

    day_english = current_time.strftime("%A")
    day_russian = days_translation.get(day_english, day_english)

    return formatted_time.replace(day_english, day_russian)


def _messages_to_input_text(messages: List[Dict[str, str]]) -> str:
    """
    Преобразует историю сообщений (формата Chat API) в плоский текст
    для передачи в Responses API через поле `input`.
    """
    parts: List[str] = []
    for item in messages or []:
        role = str(item.get("role") or "user").strip()
        content = str(item.get("content") or "").strip()
        if not content:
            continue
        if role not in ("system", "user", "assistant"):
            role = "user"
        parts.append(f"{role}:\n{content}")
    return "\n\n".join(parts).strip()


async def generate_response(
    messages: List[Dict[str, str]], system_prompt: Optional[str] = None
) -> str:
    """
    Генерирует ответ с использованием OpenAI Responses API.

    Args:
        messages: История сообщений (role/content)
        system_prompt: Системная инструкция (instructions)

    Returns:
        Строка с ответом модели
    """
    try:
        instructions = system_prompt.strip() if system_prompt else None
        input_text = _messages_to_input_text(messages)

        kwargs: Dict[str, object] = {
            "model": OPENAI_MODEL,
            "input": input_text or "",
        }
        if instructions:
            kwargs["instructions"] = instructions

        # Маппинг параметров: max_tokens → max_output_tokens
        max_output_tokens = OPENAI_PARAMS.get("max_output_tokens")
        if not isinstance(max_output_tokens, int):
            legacy_max = OPENAI_PARAMS.get("max_tokens")
            if isinstance(legacy_max, int):
                max_output_tokens = legacy_max
        if isinstance(max_output_tokens, int) and max_output_tokens > 0:
            kwargs["max_output_tokens"] = max_output_tokens

        # Температура может быть не поддержана некоторыми моделями
        temperature = OPENAI_PARAMS.get("temperature")
        used_temperature = False
        if isinstance(temperature, (int, float)):
            kwargs["temperature"] = float(temperature)
            used_temperature = True

        def _call_sync():
            nonlocal used_temperature
            try:
                return client.responses.create(**kwargs)
            except Exception as exc:
                msg = str(exc)
                if used_temperature and "Unsupported parameter" in msg and "'temperature'" in msg:
                    kwargs.pop("temperature", None)
                    used_temperature = False
                    return client.responses.create(**kwargs)
                raise

        response = await asyncio.to_thread(_call_sync)

        # Предпочтительный способ получения текста
        text = getattr(response, "output_text", None)
        if isinstance(text, str) and text.strip():
            return text.strip()

        # Резервный способ: попытка извлечения текста из структуры output
        try:
            output = getattr(response, "output", None)
            if isinstance(output, list):
                for item in output:
                    content = getattr(item, "content", None)
                    if isinstance(content, list):
                        for c in content:
                            t = getattr(c, "text", None)
                            if isinstance(t, str) and t.strip():
                                return t.strip()
        except Exception:
            pass

        logger.error("Пустой ответ текста от OpenAI Responses API")
        return "Извините, произошла ошибка при обработке запроса."
    except Exception as e:
        logger.error(f"Ошибка при запросе к OpenAI Responses API: {e}")
        return "Извините, произошла ошибка при обработке запроса. Пожалуйста, повторите попытку позже."


async def get_vibe_checker_response(user_messages: List[Dict[str, str]]) -> str:
    """
    Получает ответ от Vibe Checker с временным контекстом.

    Args:
        user_messages: История сообщений пользователя

    Returns:
        Строка с ответом Vibe Checker
    """
    time_context = get_novosibirsk_time()
    enhanced_system_prompt = f"{SYSTEM_PROMPT}\n\n{time_context}"
    return await generate_response(user_messages, enhanced_system_prompt)