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
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


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


def _format_profile_changes(applied: Dict[str, Any]) -> str:
    """Читаемое резюме обновлений профиля (для передачи ассистенту)."""
    if not isinstance(applied, dict) or not applied:
        return ""
    sex_map = {"m": "мужской", "f": "женский"}
    act_map = {"low": "низкая", "medium": "средняя", "high": "высокая"}
    goal_map = {"lose": "снижение", "maintain": "поддержание", "gain": "набор"}

    parts: list[str] = []
    if applied.get("sex") in sex_map:
        parts.append(f"пол {sex_map[applied['sex']]}")
    if isinstance(applied.get("age"), (int, float)):
        parts.append(f"возраст {int(applied['age'])}")
    if isinstance(applied.get("height_cm"), (int, float)):
        parts.append(f"рост {int(applied['height_cm'])} см")
    if isinstance(applied.get("weight_kg"), (int, float)):
        parts.append(f"вес {float(applied['weight_kg']):.2f} кг")
    if applied.get("activity") in act_map:
        parts.append(f"активность {act_map[applied['activity']]}")
    if applied.get("goal") in goal_map:
        parts.append(f"цель — {goal_map[applied['goal']]}")

    if not parts:
        return ""
    return "Профиль обновлён: " + ", ".join(parts)


def _print_reply(reply: str) -> None:
    """Безопасная печать длинного ответа по кускам, чтобы не казалось, что он «обрезан»."""
    print("Ассистент:")
    if not isinstance(reply, str) or not reply:
        print("[assistant] Пустой ответ")
        return
    chunk_size = 1000
    for i in range(0, len(reply), chunk_size):
        print(reply[i:i + chunk_size], flush=True)


def _format_profile_for_system(profile: Dict[str, Any]) -> str:
    """Формирует краткий системный контекст из актуального profile.json."""
    if not isinstance(profile, dict) or not profile:
        return ""
    sex_map = {"m": "мужской", "f": "женский"}
    act_map = {"low": "низкая", "medium": "средняя", "high": "высокая"}
    goal_map = {"lose": "снижение", "maintain": "поддержание", "gain": "набор"}
    parts: list[str] = []
    if profile.get("sex") in sex_map:
        parts.append(f"пол {sex_map[profile['sex']]}")
    if isinstance(profile.get("age"), (int, float)) and int(profile["age"]) > 0:
        parts.append(f"возраст {int(profile['age'])}")
    if isinstance(profile.get("height_cm"), (int, float)) and int(profile["height_cm"]) > 0:
        parts.append(f"рост {int(profile['height_cm'])} см")
    if isinstance(profile.get("weight_kg"), (int, float)) and float(profile["weight_kg"]) > 0:
        parts.append(f"вес {float(profile['weight_kg']):.2f} кг")
    if profile.get("activity") in act_map:
        parts.append(f"активность {act_map[profile['activity']]}")
    if profile.get("goal") in goal_map:
        parts.append(f"цель — {goal_map[profile['goal']]}")
    if not parts:
        return ""
    return "Профиль пользователя (кратко): " + ", ".join(parts)


# ===== Инструмент (Function Calling) для сохранения профиля =====
# ВАЖНО: имя/описание/параметры — на верхнем уровне объекта tools
PROFILE_SAVE_TOOL = [
    {
        "type": "function",
        "name": "save_profile_fields",
        "description": (
            "Сохранить (частично обновить) профиль пользователя. "
            "Работай ТОЛЬКО с текущим сообщением: передавай в функцию лишь те ключи, "
            "которые пользователь ЯВНО назвал в тексте. Ничего не выдумывай и не переноси "
            "старые значения из profile.json. Если новых данных нет — не вызывай функцию. "
            "Поля: sex ('m'|'f'), age (int), height_cm (int), weight_kg (number), "
            "activity ('low'|'medium'|'high'), goal ('lose'|'maintain'|'gain')."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "sex": {"type": "string", "enum": ["m", "f"]},
                "age": {"type": "integer"},
                "height_cm": {"type": "integer"},
                "weight_kg": {"type": "number"},
                "activity": {"type": "string", "enum": ["low", "medium", "high"]},
                "goal": {"type": "string", "enum": ["lose", "maintain", "gain"]},
            },
            "additionalProperties": False,
        },
    }
]


def _load_profile() -> Dict[str, Any]:
    """Загружает профиль из JSON; если файла нет — возвращает пустой словарь."""
    logger.debug("load_profile: path=%s exists=%s", str(PROFILE_PATH), PROFILE_PATH.exists())
    if PROFILE_PATH.exists():
        try:
            with PROFILE_PATH.open("r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    logger.debug("load_profile: loaded keys=%s", list(data.keys()))
                    return data
        except Exception:
            logger.exception("load_profile: failed to read/parse JSON")
            pass
    return {}


def _save_profile(profile: Dict[str, Any]) -> None:
    """Сохраняет профиль в JSON c аккуратной индентацией."""
    logger.debug("save_profile: keys=%s", list(profile.keys()))
    PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with PROFILE_PATH.open("w", encoding="utf-8") as f:
        json.dump(profile, f, ensure_ascii=False, indent=2)
    logger.info("save_profile: written to %s", str(PROFILE_PATH))


def save_profile_fields_to_json(updates: Dict[str, Any]) -> Dict[str, Any]:
    """Применяет частичное обновление к profile.json и возвращает применённые поля."""
    if not isinstance(updates, dict) or not updates:
        logger.debug("save_profile_fields_to_json: empty updates")
        return {}
    allowed_keys = {"sex", "age", "height_cm", "weight_kg", "activity", "goal"}
    raw_safe = {k: v for k, v in updates.items() if k in allowed_keys}
    if not raw_safe:
        logger.debug("save_profile_fields_to_json: no allowed keys in updates=%s", list(updates.keys()))
        return {}

    # Фильтрация: применяем ТОЛЬКО реально изменённые и валидные значения.
    current = _load_profile()
    applied: Dict[str, Any] = {}

    enums = {
        "sex": {"m", "f"},
        "activity": {"low", "medium", "high"},
        "goal": {"lose", "maintain", "gain"},
    }
    numeric_fields = {"age", "height_cm", "weight_kg"}

    for key, new_val in raw_safe.items():
        old_val = current.get(key)

        # Пропуски и нули
        if new_val is None:
            logger.debug("skip %s: value is None", key)
            continue
        if isinstance(new_val, str) and not new_val.strip():
            logger.debug("skip %s: empty string", key)
            continue

        # Валидация enum-полей
        if key in enums:
            s = str(new_val).lower()
            if s not in enums[key]:
                logger.debug("skip %s: invalid enum value=%s", key, new_val)
                continue
            if isinstance(old_val, str) and old_val.lower() == s:
                logger.debug("skip %s: unchanged (enum) old=%s new=%s", key, old_val, s)
                continue
            applied[key] = s
            continue

        # Валидация числовых полей (игнорируем нереалистичные/нулевые)
        if key in numeric_fields:
            try:
                num_val = float(new_val)
            except Exception:
                logger.debug("skip %s: not a number value=%s", key, new_val)
                continue
            # простые нижние пороги адекватности
            if key == "age" and (num_val <= 0 or num_val > 120):
                logger.debug("skip %s: out-of-range age=%s", key, num_val)
                continue
            if key == "height_cm" and (num_val <= 80 or num_val > 250):
                logger.debug("skip %s: out-of-range height_cm=%s", key, num_val)
                continue
            if key == "weight_kg" and (num_val <= 20 or num_val > 400):
                logger.debug("skip %s: out-of-range weight_kg=%s", key, num_val)
                continue
            # сравнение с текущим
            try:
                old_num = float(old_val) if old_val is not None else None
            except Exception:
                old_num = None
            if old_num is not None:
                # Для веса учтём маленький допуск
                if key == "weight_kg" and abs(old_num - num_val) < 1e-6:
                    logger.debug("skip %s: unchanged old=%s new=%s", key, old_num, num_val)
                    continue
                if key in ("age", "height_cm") and int(round(old_num)) == int(round(num_val)):
                    logger.debug("skip %s: unchanged old=%s new=%s", key, old_num, num_val)
                    continue
            # Записываем приведённые типы
            if key in ("age", "height_cm"):
                applied[key] = int(round(num_val))
            elif key == "weight_kg":
                applied[key] = float(num_val)
            continue

        # Прочие строки (сейчас таких нет) — сравнение и запись, если изменилось
        if old_val == new_val:
            logger.debug("skip %s: unchanged old=%s new=%s", key, old_val, new_val)
            continue
        applied[key] = new_val

    if not applied:
        logger.info("save_profile_fields_to_json: nothing to apply after filtering; raw_keys=%s", list(raw_safe.keys()))
        return {}

    # Применяем только изменённые
    current.update(applied)
    _save_profile(current)
    logger.info("save_profile_fields_to_json: applied=%s", applied)
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
        "Извлекай только явно названные поля (sex, age, height_cm, weight_kg, activity, goal). "
        "Запрещено придумывать/делать выводы или переносить прошлые значения. "
        "Передавай в save_profile_fields ТОЛЬКО те ключи, которые буквально присутствуют в тексте пользователя, с теми же значениями. "
        "Если новых данных нет — НЕ вызывай функцию."
    )

    try:
        logger.info("tools: start parse; model=%s", MODEL)
        resp = client.responses.create(
            model=MODEL,
            instructions=instructions,
            input=text,
            tools=PROFILE_SAVE_TOOL,
            tool_choice="auto",
        )
    except Exception as exc:
        # На учебном примере просто печатаем проблему и выходим без обновления
        print(f"[tools] Ошибка вызова модели: {exc}")
        logger.exception("tools: call failed")
        return {}

    # Универсальный парсинг function_call из Responses API
    applied: Dict[str, Any] = {}
    saw_tool = False
    saw_args = False
    for item in getattr(resp, "output", []) or []:
        item_type = getattr(item, "type", None)
        if item_type in ("function_call", "tool_call"):
            name = getattr(item, "name", "")
            if name == "save_profile_fields":
                saw_tool = True
                args_raw = getattr(item, "arguments", "") or getattr(item, "input", "")
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

        # Основной вызов (пробуем форсировать текстовую модальность, если модель поддерживает)
        try:
            logger.info("reply: create; model=%s; with_profile=%s; user_len=%d", MODEL, bool(profile_ctx), len(user_text or ""))
            resp = client.responses.create(
                model=MODEL,
                instructions=instructions,
                input=user_text or "",
                max_output_tokens=600,
                response_format={"type": "text"},
            )
        except Exception:
            # Фолбэк без response_format
            logger.warning("reply: fallback create without response_format")
            resp = client.responses.create(
                model=MODEL,
                instructions=instructions,
                input=user_text or "",
                max_output_tokens=600,
            )

        text = getattr(resp, "output_text", None)
        if isinstance(text, str) and text.strip():
            logger.info("reply: parsed via output_text; len=%d", len(text.strip()))
            return text.strip()

        # Резервный парсинг
        for item in getattr(resp, "output", []) or []:
            content = getattr(item, "content", None)
            if isinstance(content, list):
                for c in content:
                    t = getattr(c, "text", None)
                    if isinstance(t, str) and t.strip():
                        logger.info("reply: parsed via content[]; len=%d", len(t.strip()))
                        return t.strip()

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


