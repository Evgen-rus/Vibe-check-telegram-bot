"""
Минимальный учебный пример «ИИ‑консультанта» с auto‑update профиля через Tools (Function Calling).

Что показывает скрипт:
1) Короткий системный промпт (основная роль ассистента)
2) Одноразовый вызов модели с tools для извлечения и сохранения полей профиля
3) Сохранение профиля в JSON‑файл (без БД для простоты)
4) Простой REPL‑чат в консоли

Запуск:
  - Установи переменную окружения OPENAI_API_KEY
  - python examples/minimal_ai_consultant.py

Как пользоваться:
  - Пиши обычным текстом, например: «Мне 31, рост 178, вес 82. Цель — снизить. Активность — средняя.»
  - Скрипт попытается извлечь поля профиля и сохранить их в profile.json
  - Затем ассистент выдаст короткий ответ по теме

Примечание:
  - Используется Responses API: instructions + input
  - Инструмент описан в формате tools (name/description/parameters на верхнем уровне)

Мини-пример вложенного вида (эквивалент вашего инструмента):
{
  "type": "function",
  "function": {
    "name": "save_profile_fields",
    "description": "Сохраняй только явно упомянутые поля профиля...",
    "strict": true,
    "parameters": {
      "type": "object",
      "properties": {
        "sex": { "type": ["string","null"], "enum": ["m","f"] },
        "age": { "type": ["integer","null"] },
        "height_cm": { "type": ["integer","null"] },
        "weight_kg": { "type": ["number","null"] },
        "activity": { "type": ["string","null"], "enum": ["low","medium","high"] },
        "goal": { "type": ["string","null"], "enum": ["lose","maintain","gain"] }
      },
      "required": ["sex","age","height_cm","weight_kg","activity","goal"],
      "additionalProperties": false
    }
  }
}
Коротко:
Ваш стиль соответствует одному из вариантов из доков и может работать.
Для большей совместимости с Responses/Assistants чаще используют вложенную форму function: { ... }.
Если всё работает — можно не трогать. Если инструмент не вызывается — попробуйте вложенный стиль.

У «рассуждающих» моделей (GPT‑5, o4‑mini) ответ может содержать не только вызов инструмента, но и отдельные элементы типа reasoning (внутренние рассуждения в структурированном виде).
### Что это значит
- Когда вы возвращаете в модель результаты работы инструмента (тот самый «шаг с отправкой вывода инструментов»), эти reasoning‑элементы из предыдущего ответа тоже нужно передать обратно вместе с выводами инструмента.
- Это нужно, чтобы модель сохранила контекст хода рассуждений между шагами и корректно продолжила решение.

### Когда это применяется
- Только при классическом «tool loop»: модель → tool_call → вы исполняете инструмент → отправляете tool outputs обратно в ту же «сессию ответа» для продолжения.
- Если вы не отправляете tool outputs назад в ту же сессию (как в вашем скрипте — вы просто локально сохраняете профиль и делаете отдельный новый вызов модели), это требование на вас не распространяется.

### Почему важно
- Без повторной передачи reasoning‑элементов модель может «потерять нить» рассуждений и дать более слабый финальный ответ.
- Некоторые SDK/эндпоинты для reasoning‑моделей ожидают этот протокол, иначе поведение может быть непредсказуемым.

### Как выглядит на практике (упрощённо)
- Шаг 1: модель вернула смешанный ответ: reasoning + tool_call.
- Шаг 2: вы исполнили инструмент и делаете следующий запрос/шаг, в котором передаёте:
  - результаты инструмента;
  - те самые reasoning‑элементы из шага 1.
- Шаг 3: модель продолжает рассуждение, опираясь на оба — reasoning и результаты инструмента.

В вашем `minimal_ai_consultant.py` вы не запускаете «многошаговую» сессию с обратной подачей tool outputs; вы делаете один вызов для парсинга (и сохраняете локально), а затем второй независимый вызов для ответа. Поэтому требование «передать reasoning обратно» вам сейчас не нужно.

- Если захотите переделать под настоящий tool‑loop (с подачей результатов инструмента обратно в ту же сессию), подскажу, как правильно прокинуть reasoning вместе с выводом инструмента.

- Важное: reasoning‑элементы — это структурированная часть ответа для самой модели, а не то, что нужно показывать пользователю. Обычно их не логируют публично.
"""

from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

from openai import OpenAI
from dotenv import load_dotenv
import logging


# ===== Загрузка .env и настройка модели =====
load_dotenv()  # ищет .env в текущей директории и выше
MODEL = os.getenv("OPENAI_MODEL") or os.getenv("MODEL") or "gpt-5-mini"


# ===== Пути =====
THIS_DIR = Path(__file__).resolve().parent
PROFILE_PATH = THIS_DIR / "profile.json"
LOG_FILE = THIS_DIR / "minimal_ai_consultant.log"


# ===== Инициализация клиента =====
# Сетевые настройки: мягкие таймауты и немного ретраев, чтобы не зависать при проблемах сети/API
REQUEST_TIMEOUT_SEC = float(os.getenv("OPENAI_TIMEOUT", "200"))
REQUEST_MAX_RETRIES = int(os.getenv("OPENAI_MAX_RETRIES", "1"))

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    timeout=REQUEST_TIMEOUT_SEC,
    max_retries=REQUEST_MAX_RETRIES,
)


# ===== Логирование в файл examples/minimal_ai_consultant.log =====
logger = logging.getLogger("minimal_ai_consultant")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    # Лог первой строки — старт
    logger.info("Logger initialized; log file: %s", str(LOG_FILE))

# Управление подробностью валидации/чтения (по умолчанию минимум шума)
VERBOSE_VALIDATION = False

def _vdebug(msg: str, *args: Any) -> None:
    if VERBOSE_VALIDATION:
        try:
            logger.debug(msg, *args)
        except Exception:
            pass


# ===== Основной (короткий) промпт консультанта =====
SYSTEM_PROMPT = (
    "Ты — доброжелательный и краткий консультант по питанию/здоровью. "
    "Всегда отвечай обычным текстом (2–4 предложения). Если профиль содержит данные — учитывай их."
)


def _log_debug(stage: str, info: str) -> None:
    """Короткое понятное логирование для диагностики (только при нестандартных ситуациях)."""
    try:
        print(f"[debug] {stage}: {info}")
        logger.debug("%s: %s", stage, info)
    except Exception:
        pass


def _print_reply(reply: str) -> None:
    """Безопасная печать длинного ответа по кускам, чтобы не казалось, что он «обрезан»."""
    print("Ассистент:")
    if not isinstance(reply, str):
        print("[assistant] Некорректный тип ответа")
        return
    # Печатаем ответ полностью, без чанков и .strip()
    print(reply if reply is not None else "")


def _debug_json(title: str, obj: Any) -> None:
    """Печать и логирование объекта как JSON (для обучения/диагностики)."""
    try:
        serialized = json.dumps(obj, ensure_ascii=False, indent=2)
    except Exception:
        serialized = repr(obj)
    _log_debug(title, serialized)


def _summarize_response_for_log(resp: Any) -> Dict[str, Any]:
    """Безопасная сводка ответа модели для логов: output_text + разбор output[]."""
    summary: Dict[str, Any] = {
        "output_text": getattr(resp, "output_text", None),
        "output": []
    }
    try:
        for item in getattr(resp, "output", []) or []:
            item_type = getattr(item, "type", None)
            entry: Dict[str, Any] = {"type": item_type}
            if item_type in ("function_call", "tool_call"):
                entry["name"] = getattr(item, "name", "")
                entry["arguments"] = getattr(item, "arguments", None) or getattr(item, "input", None)
            content = getattr(item, "content", None)
            if isinstance(content, list):
                texts: List[str] = []
                for c in content:
                    t = getattr(c, "text", None)
                    if isinstance(t, str):
                        texts.append(t)
                if texts:
                    entry["texts"] = texts
            summary["output"].append(entry)
    except Exception:
        pass
    return summary


def _log_raw_response(title: str, resp: Any) -> None:
    """Логирование «сырого» ответа модели максимально близко к 1:1.
    Пытаемся использовать model_dump_json()/model_dump(), иначе to_dict()/repr()."""
    try:
        # Pydantic v2 стиль
        dump_json = getattr(resp, "model_dump_json", None)
        if callable(dump_json):
            try:
                raw_json = dump_json(indent=2)
                _log_debug(title, raw_json)
                return
            except Exception:
                pass
        dump = getattr(resp, "model_dump", None)
        if callable(dump):
            try:
                raw = dump()
                _debug_json(title, raw)
                return
            except Exception:
                pass
        to_dict = getattr(resp, "to_dict", None)
        if callable(to_dict):
            try:
                raw = to_dict()
                _debug_json(title, raw)
                return
            except Exception:
                pass
        # Фолбэк
        _log_debug(title, repr(resp))
    except Exception:
        try:
            _log_debug(title, repr(resp))
        except Exception:
            pass


def _format_profile_for_system(profile: Dict[str, Any]) -> str:
    """Формирует краткий системный контекст из актуального profile.json."""
    if not isinstance(profile, dict) or not profile:
        return ""
    sex_map = {"m": "мужской", "f": "женский"}
    parts: list[str] = []
    if profile.get("sex") in sex_map:
        parts.append(f"пол {sex_map[profile['sex']]}")
    if isinstance(profile.get("age"), (int, float)) and int(profile["age"]) > 0:
        parts.append(f"возраст {int(profile['age'])}")
    if isinstance(profile.get("weight_kg"), (int, float)) and float(profile["weight_kg"]) > 0:
        parts.append(f"вес {float(profile['weight_kg']):.2f} кг")
    if not parts:
        return ""
    return "Профиль пользователя (кратко): " + ", ".join(parts)


# ===== Инструмент (Function Calling) для сохранения профиля =====
# ВАЖНО: имя/описание/параметры — на верхнем уровне объекта tools
PROFILE_SAVE_TOOL = [
    {
        "type": "function",
        "name": "save_profile_fields",
        "description": "Сохраняй только явно упомянутые в сообщении пользователя поля профиля. Для перечислимых полей используй канонические значения.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "sex": {
                    "type": ["string", "null"],
                    "enum": ["m", "f"],
                    "description": "Пол: 'm' или 'f' (канонические значения)."
                },
                "age": {
                    "type": ["integer", "null"],
                    "description": "Возраст (целое число)."
                },
                "weight_kg": {
                    "type": ["number", "null"],
                    "description": "Вес в килограммах (число)."
                },
            },
            "required": ["sex", "age", "weight_kg"],
            "additionalProperties": False,
        },
    }
]


def _load_profile() -> Dict[str, Any]:
    """Загружает профиль из JSON; если файла нет — возвращает пустой словарь."""
    _vdebug("load_profile: path=%s exists=%s", str(PROFILE_PATH), PROFILE_PATH.exists())
    if PROFILE_PATH.exists():
        try:
            with PROFILE_PATH.open("r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    _vdebug("load_profile: loaded keys=%s", list(data.keys()))
                    return data
        except Exception:
            logger.exception("load_profile: failed to read/parse JSON")
            pass
    return {}


def _save_profile(profile: Dict[str, Any]) -> None:
    """Сохраняет профиль в JSON c аккуратной индентацией."""
    _vdebug("save_profile: keys=%s", list(profile.keys()))
    PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with PROFILE_PATH.open("w", encoding="utf-8") as f:
        json.dump(profile, f, ensure_ascii=False, indent=2)
    logger.info("save_profile: written to %s", str(PROFILE_PATH))


def save_profile_fields_to_json(updates: Dict[str, Any]) -> Dict[str, Any]:
    """Применяет частичное обновление к profile.json и возвращает применённые поля."""
    if not isinstance(updates, dict) or not updates:
        _vdebug("save_profile_fields_to_json: empty updates")
        return {}
    allowed_keys = {"sex", "age", "weight_kg"}
    raw_safe = {k: v for k, v in updates.items() if k in allowed_keys}
    if not raw_safe:
        _vdebug("save_profile_fields_to_json: no allowed keys in updates=%s", list(updates.keys()))
        return {}

    # Фильтрация: применяем ТОЛЬКО реально изменённые и валидные значения.
    current = _load_profile()
    applied: Dict[str, Any] = {}

    enums = {
        "sex": {"m", "f"},
    }
    numeric_fields = {"age", "weight_kg"}

    for key, new_val in raw_safe.items():
        old_val = current.get(key)

        # Пропуски и нули
        if new_val is None:
            _vdebug("skip %s: value is None", key)
            continue
        if isinstance(new_val, str) and not new_val.strip():
            _vdebug("skip %s: empty string", key)
            continue

        # Валидация enum-полей
        if key in enums:
            s = str(new_val).lower()
            if s not in enums[key]:
                _vdebug("skip %s: invalid enum value=%s", key, new_val)
                continue
            if isinstance(old_val, str) and old_val.lower() == s:
                _vdebug("skip %s: unchanged (enum) old=%s new=%s", key, old_val, s)
                continue
            applied[key] = s
            continue

        # Валидация числовых полей (игнорируем нереалистичные/нулевые)
        if key in numeric_fields:
            try:
                num_val = float(new_val)
            except Exception:
                _vdebug("skip %s: not a number value=%s", key, new_val)
                continue
            # Упростим: без порогов адекватности в обучающем режиме
            # сравнение с текущим
            try:
                old_num = float(old_val) if old_val is not None else None
            except Exception:
                old_num = None
            if old_num is not None:
                # Для веса учтём маленький допуск
                if key == "weight_kg" and abs(old_num - num_val) < 1e-6:
                    _vdebug("skip %s: unchanged old=%s new=%s", key, old_num, num_val)
                    continue
                if key == "age" and int(round(old_num)) == int(round(num_val)):
                    _vdebug("skip %s: unchanged old=%s new=%s", key, old_num, num_val)
                    continue
            # Записываем приведённые типы
            if key == "age":
                applied[key] = int(round(num_val))
            elif key == "weight_kg":
                applied[key] = float(num_val)
            continue

        # Прочие строки (сейчас таких нет) — сравнение и запись, если изменилось
        if old_val == new_val:
            _vdebug("skip %s: unchanged old=%s new=%s", key, old_val, new_val)
            continue
        applied[key] = new_val

    if not applied:
        logger.info("tools.parsed: nothing_applied; raw_keys=%s", list(raw_safe.keys()))
        return {}

    # Применяем только изменённые
    current.update(applied)
    _save_profile(current)
    logger.info("tools.parsed: applied=%s", applied)
    return applied


def maybe_update_profile_from_text(text: str) -> Dict[str, Any]:
    """
    Делает ОДИН короткий вызов модели с tools. Если пользователь в тексте
    явно назвал новые поля профиля — модель вернёт function_call и аргументы.
    Мы применим их к profile.json. Возвращает dict применённых полей (или пустой dict).
    """
    if not text:
        return {}

    instructions = (
        "Ты — парсер профиля. Работай ТОЛЬКО с текущим сообщением пользователя. "
        "Твой ответ должен быть коротким, поэтому не используй никаких слов, кроме значений полей."
        "Извлекай только явно названные поля (sex, age, weight_kg). "
        "Запрещено придумывать/делать выводы или переносить прошлые значения. "
        "Передавай в save_profile_fields ТОЛЬКО те ключи, которые присутствуют в тексте пользователя. "
        "Нормализуй поле sex к канонам: 'm'|'f'. "
        "Если новых данных нет — НЕ вызывай функцию."
    )

    try:
        logger.info("tools: start parse; model=%s", MODEL)
        # Полный лог запроса к модели (для обучения)
        _debug_json(
            "tools.request",
            {
                "model": MODEL,
                "instructions": instructions,
                "input": text,
                "tools": PROFILE_SAVE_TOOL,
                "tool_choice": "auto",
            },
        )
        resp = client.responses.create(
            model=MODEL,
            instructions=instructions,
            input=text,
            tools=PROFILE_SAVE_TOOL,
            tool_choice="auto",
        )
        # Логируем «сырой» ответ модели для обучения
        _log_raw_response("tools.response.raw", resp)
    except Exception as exc:
        # На учебном примере просто печатаем проблему и выходим без обновления
        print(f"[tools] Ошибка вызова модели: {exc}")
        logger.exception("tools: call failed")
        return {}

    # Универсальный парсинг function_call из Responses API
    applied: Dict[str, Any] = {}
    saw_tool = False
    saw_args = False
    # (сводка ответа можно включить через VERBOSE_VALIDATION при необходимости)

    for item in getattr(resp, "output", []) or []:
        item_type = getattr(item, "type", None)
        if item_type in ("function_call", "tool_call"):
            name = getattr(item, "name", "")
            if name == "save_profile_fields":
                saw_tool = True
                args_raw = getattr(item, "arguments", "") or getattr(item, "input", "")
                # Логируем сырые аргументы вызова инструмента
                _debug_json("tools.function_call.arguments_raw", args_raw)
                try:
                    args = json.loads(args_raw) if isinstance(args_raw, str) else (args_raw or {})
                except Exception:
                    args = {}
                if isinstance(args, dict) and args:
                    saw_args = True
                    logger.info("tools: function_call received; keys=%s", list(args.keys()))
                    applied = save_profile_fields_to_json(args)
                break
    if not saw_tool:
        _log_debug("tools", "no function_call emitted")
        logger.info("tools: no function_call emitted")
    elif not saw_args:
        _log_debug("tools", "function_call without valid arguments")
        logger.info("tools: function_call without valid arguments")
    else:
        _log_debug("tools", f"profile updated: {list(applied.keys())}")
        logger.info("tools: profile updated; applied_keys=%s", list(applied.keys()))
    return applied


def generate_assistant_reply(user_text: str) -> str:
    """
    Генерирует короткий ответ ассистента только из:
      - системного промпта + краткого профиля (из profile.json)
      - последнего сообщения пользователя (user_text)
    """
    try:
        profile = _load_profile()
        profile_ctx = _format_profile_for_system(profile)
        instructions = SYSTEM_PROMPT if not profile_ctx else f"{SYSTEM_PROMPT}\n\n{profile_ctx}"

        logger.info("reply: create; model=%s; with_profile=%s; user_len=%d", MODEL, bool(profile_ctx), len(user_text or ""))
        # Один вызов без response_format — по логам он и так используется
        _debug_json(
            "reply.request",
            {
                "model": MODEL,
                "instructions": instructions,
                "input": user_text or "",
                "max_output_tokens": 5000,
            },
        )
        resp = client.responses.create(
            model=MODEL,
            instructions=instructions,
            input=user_text or "",
            max_output_tokens=5000,
        )
        _log_raw_response("reply.response.raw", resp)

        # (сводку ответа можно включить через VERBOSE_VALIDATION при необходимости)

        text = getattr(resp, "output_text", None)
        if isinstance(text, str) and len(text) > 0:
            logger.info("reply: parsed via output_text; len=%d", len(text))
            return text

        # Резервный парсинг: собираем все текстовые куски в один блок
        collected_texts: List[str] = []
        for item in getattr(resp, "output", []) or []:
            content = getattr(item, "content", None)
            if isinstance(content, list):
                for c in content:
                    t = getattr(c, "text", None)
                    if isinstance(t, str) and len(t) > 0:
                        collected_texts.append(t)
        if collected_texts:
            merged = "\n\n".join(collected_texts)
            logger.info("reply: parsed via merged content[]; len=%d", len(merged))
            return merged

        # Если сюда дошли — модель не вернула пригодный текст
        out_len = len(getattr(resp, "output", []) or [])
        _log_debug("reply", f"empty output_text; output_items={out_len}")
        logger.warning("reply: empty output_text; output_items=%d", out_len)
    except Exception as exc:
        logger.exception("reply: create failed")
        return f"[assistant] Ошибка вызова модели: {exc}"
    return "[assistant] Не удалось получить ответ"


def main() -> None:
    print("Мини‑чат с авто‑обновлением профиля через Tools. Напиши /exit для выхода.")
    print("Профиль сохраняется в:", PROFILE_PATH)
    logger.info("app: start; model=%s; profile=%s", MODEL, str(PROFILE_PATH))

    while True:
        try:
            user_text = input("\nВы: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nВыход.")
            logger.info("app: exit (interrupt)")
            break

        if not user_text or user_text.lower() == "/exit":
            print("Выход.")
            logger.info("app: exit (command)")
            break

        # Короткий лог полного сообщения пользователя
        logger.info("user.input: %s", user_text)
        # Дополнительно — длина и превью (оставим на случай удобного сканирования)
        logger.info("user: input len=%d; text_preview=%s", len(user_text), user_text[:120])
        # 1) Одноразовый парсинг профиля (tools)
        applied = maybe_update_profile_from_text(user_text)
        if applied:
            # Покажем, что именно изменилось
            print("Ассистент: ✅ Профиль обновлён:")
            print(json.dumps(applied, ensure_ascii=False))
            logger.info("app: printed profile updates; keys=%s", list(applied.keys()))

        # 2) Основной короткий ответ ассистента (используем только профиль из файла + последний ввод)
        reply = generate_assistant_reply(user_text)
        _print_reply(reply)
        logger.info("app: reply printed; is_error=%s", reply.startswith("[assistant]"))


if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("Требуется переменная окружения OPENAI_API_KEY")
    else:
        main()


