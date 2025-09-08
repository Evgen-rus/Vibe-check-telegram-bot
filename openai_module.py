"""
Модуль для взаимодействия с OpenAI Responses API (асинхронно).
"""

from typing import List, Dict, Optional, Any
from datetime import datetime
import json
import pytz
from openai import AsyncOpenAI
from config import OPENAI_API_KEY, OPENAI_MODEL, OPENAI_PARAMS, logger, INCLUDE_REMINDERS_IN_PROMPT, INCLUDE_PROFILE_IN_PROMPT, get_user_tz, MOSCOW_USERS
from storage import storage
from prompts import SYSTEM_PROMPT

# Настройка клиента OpenAI (асинхронный Responses API)
client = AsyncOpenAI(api_key=OPENAI_API_KEY)


def get_time_for_user(user_id: Optional[int]) -> str:
    """
    Получает локальное время пользователя для системного контекста.

    Returns:
        Строка с текущим временем в формате для AI
    """
    tz = get_user_tz(int(user_id)) if user_id is not None else pytz.timezone("Asia/Novosibirsk")
    current_time = datetime.now(tz)

    city_label = "Москва" if (user_id is not None and int(user_id) in MOSCOW_USERS) else "Новосибирск"
    formatted_time = current_time.strftime(
        f"Текущее время: %d.%m.%Y, %H:%M, %A, {city_label}"
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


def _log_token_usage(response: Any, model_name: str, context: Optional[Dict[str, Any]] = None) -> None:
    try:
        usage = getattr(response, "usage", None)
        if not usage:
            return
        # Попытка извлечь популярные поля из usage
        fields = [
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "prompt_tokens",
            "completion_tokens",
            "reasoning_tokens",
        ]
        usage_info: Dict[str, Any] = {}
        for f in fields:
            val = getattr(usage, f, None)
            if isinstance(val, (int, float)):
                usage_info[f] = int(val)
        # Если ничего не извлекли, попробуем to_dict/model_dump
        if not usage_info:
            to_dict = getattr(usage, "model_dump", None) or getattr(usage, "to_dict", None)
            if callable(to_dict):
                try:
                    dump = to_dict()
                    if isinstance(dump, dict):
                        for k in ("input_tokens", "output_tokens", "total_tokens", "prompt_tokens", "completion_tokens"):
                            if k in dump and isinstance(dump[k], (int, float)):
                                usage_info[k] = int(dump[k])
                except Exception:
                    pass
        log_parts = [f"model={model_name}"]
        if context:
            ctx_str = ", ".join(f"{k}={v}" for k, v in context.items())
            if ctx_str:
                log_parts.append(ctx_str)
        if usage_info:
            usage_str = ", ".join(f"{k}={v}" for k, v in usage_info.items())
            logger.info(f"OpenAI usage: {'; '.join(log_parts)}; {usage_str}")
    except Exception as exc:
        logger.debug(f"Не удалось залогировать токены: {exc}")


async def generate_response(
    messages: List[Dict[str, str]], system_prompt: Optional[str] = None, *, log_context: Optional[Dict[str, Any]] = None
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

        # Не передаём temperature: некоторые модели (например, GPT-5) не поддерживают этот параметр

        # Основной запрос к Responses API
        try:
            response = await client.responses.create(**kwargs)
        except Exception as exc:
            msg = str(exc)
            # Фолбэк при невалидной модели
            if any(substr in msg.lower() for substr in [
                "invalid model", "invalid model id", "model_not_found", "unknown model"
            ]):
                fallback_model = "gpt-4.1-mini"
                logger.warning(
                    f"Модель {OPENAI_MODEL} недоступна или неверна. Пробую запасную модель: {fallback_model}"
                )
                kwargs["model"] = fallback_model
                response = await client.responses.create(**kwargs)
            else:
                raise

        # Логгируем использование токенов (в логи, не пользователю)
        try:
            _log_token_usage(response, str(kwargs.get("model", OPENAI_MODEL)), log_context)
        except Exception:
            pass

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


async def get_vibe_checker_response(user_messages: List[Dict[str, str]], *, user_id: Optional[int] = None) -> str:
    """
    Получает ответ от Vibe Checker с временным контекстом и (опционально) кратким контекстом напоминаний.

    Args:
        user_messages: История сообщений пользователя
        user_id: ID пользователя для выборки напоминаний (необязателен)

    Returns:
        Строка с ответом Vibe Checker
    """
    time_context = get_time_for_user(user_id)
    reminders_block = ""
    if INCLUDE_REMINDERS_IN_PROMPT and user_id is not None:
        try:
            lines = await storage.get_compact_reminders_context(user_id, limit=5)
            if lines:
                reminders_block = "\n\nАктивные напоминания пользователя (кратко):\n- " + "\n- ".join(lines)
        except Exception as exc:
            logger.warning(f"Не удалось собрать контекст напоминаний: {exc}")
    profile_block = ""
    if INCLUDE_PROFILE_IN_PROMPT and user_id is not None:
        try:
            profile_line = await storage.get_compact_profile_context(user_id)
            if profile_line:
                profile_block = "\n\nПрофиль пользователя (кратко): " + profile_line
        except Exception as exc:
            logger.warning(f"Не удалось собрать профиль пользователя: {exc}")

    enhanced_system_prompt = f"{SYSTEM_PROMPT}\n\n{time_context}{profile_block}{reminders_block}"
    log_ctx = {"user_id": user_id, "messages": len(user_messages) if user_messages else 0}
    return await generate_response(user_messages, enhanced_system_prompt, log_context=log_ctx)


# === Tools / Function Calling: авто-сохранение профиля ===
PROFILE_SAVE_TOOL = [
    {
        "type": "function",
        "name": "save_profile_fields",
        "description": (
            "Сохранить (частично обновить) профиль пользователя. "
            "Передавай ТОЛЬКО те поля, которые пользователь явно назвал или подтвердил. "
            "Поля: sex ('m'/'f'), age (int), height_cm (int), weight_kg (float/int), "
            "activity ('low'|'medium'|'high'), goal ('lose'|'maintain'|'gain'), "
            "allergies (string), diet (string)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "sex":       {"type": "string", "enum": ["m", "f"]},
                "age":       {"type": "integer"},
                "height_cm": {"type": "integer"},
                "weight_kg": {"type": "number"},
                "activity":  {"type": "string", "enum": ["low", "medium", "high"]},
                "goal":      {"type": "string", "enum": ["lose", "maintain", "gain"]},
                "allergies": {"type": "string"},
                "diet":      {"type": "string"}
            },
            "additionalProperties": False
        },
    }
]


async def maybe_update_profile_from_text(text: str, user_id: int) -> dict:
    """
    Одноразовый вызов модели с tools: если в тексте есть НОВЫЕ данные профиля —
    модель вызовет save_profile_fields с нужными аргументами. Мы их сохраним в БД.
    Возвращает dict обновлённых полей (или пустой dict).
    """
    if not text or user_id is None:
        return {}

    instructions = (
        "Ты — парсер профиля. Если пользователь назвал новые данные профиля "
        "(sex, age, height_cm, weight_kg, activity, goal, allergies, diet), "
        "ВЫЗОВИ функцию save_profile_fields, заполняя ТОЛЬКО явно подтверждённые поля. "
        "Если новых данных нет — не вызывай функцию."
    )

    try:
        resp = await client.responses.create(
            model=OPENAI_MODEL,
            instructions=instructions,
            input=text,
            tools=PROFILE_SAVE_TOOL,
            tool_choice="auto",
        )
    except Exception as exc:
        msg = str(exc)
        logger.warning(f"Function-calling profile parse failed on model {OPENAI_MODEL}: {msg}")
        fallback_model = "gpt-4.1-mini"
        try:
            resp = await client.responses.create(
                model=fallback_model,
                instructions=instructions,
                input=text,
                tools=PROFILE_SAVE_TOOL,
                tool_choice="auto",
            )
            logger.info(f"Function-calling profile parse succeeded with fallback model {fallback_model}")
        except Exception as exc2:
            logger.error(f"Function-calling profile parse failed with fallback: {exc2}")
            return {}

    updated: Dict[str, Any] = {}
    for item in getattr(resp, "output", []) or []:
        t = getattr(item, "type", None)
        if t in ("function_call", "tool_call"):
            name = getattr(item, "name", "")
            if name == "save_profile_fields":
                args_raw = getattr(item, "arguments", "") or getattr(item, "input", "")
                try:
                    args = json.loads(args_raw) if isinstance(args_raw, str) else (args_raw or {})
                except Exception:
                    args = {}
                if isinstance(args, dict) and args:
                    await storage.set_profile_fields(user_id, **args)
                    updated = args
                    try:
                        logger.info(f"Profile updated via tool-call for user_id={user_id}: {args}")
                    except Exception:
                        pass
                break

    return updated