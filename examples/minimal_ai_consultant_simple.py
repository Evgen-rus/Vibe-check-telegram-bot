"""
Мини‑версия "ИИ‑консультанта" для обучения Function Calling (OpenAI). 

Цель: показать самый простой рабочий цикл:
1) Один вызов модели с tools -> модель вызывает функцию `save_profile_fields` с аргументами
2) Мы обновляем `profile.json`
3) Второй простой вызов модели -> короткий ответ с учётом профиля

Что важно понять:
- Function Calling = модель решает, когда и с какими аргументами звать вашу функцию.
- Вы описываете схему параметров (JSON Schema) — модель её придерживается.
- В этом файле НЕТ многошагового "tool loop". Мы не отправляем обратно результаты функции в ту же сессию ответа. 
  Для обучения этого достаточно. Позже можно усложнить.

Запуск:
  - Установите переменную окружения OPENAI_API_KEY
  - Опционально: OPENAI_TIMEOUT (сек, по умолчанию 20), OPENAI_MAX_RETRIES (по умолчанию 1)
  - python examples/minimal_ai_consultant_simple.py
"""

from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Dict, Any

from dotenv import load_dotenv
from openai import OpenAI


# === Настройки окружения и модели ===
load_dotenv()
MODEL = os.getenv("OPENAI_MODEL") or os.getenv("MODEL") or "gpt-5-mini"
REQUEST_TIMEOUT_SEC = float(os.getenv("OPENAI_TIMEOUT", "200"))
REQUEST_MAX_RETRIES = int(os.getenv("OPENAI_MAX_RETRIES", "1"))


# === Пути ===
THIS_DIR = Path(__file__).resolve().parent
PROFILE_PATH = THIS_DIR / "profile.json"


# === Клиент OpenAI с мягкими таймаутами/ретраями ===
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    timeout=REQUEST_TIMEOUT_SEC,
    max_retries=REQUEST_MAX_RETRIES,
)


# === Инструмент (Function Calling) ===
# Используем верхнеуровневый формат tools, как в доках OpenAI
PROFILE_SAVE_TOOL = [
    {
        "type": "function",
        "name": "save_profile_fields",
        "description": "Сохраняй только явно названные в сообщении пользователя поля: sex(m|f), age(int), weight_kg(number).",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "sex": {"type": ["string", "null"], "enum": ["m", "f"]},
                "age": {"type": ["integer", "null"]},
                "weight_kg": {"type": ["number", "null"]},
            },
            "required": ["sex", "age", "weight_kg"],
            "additionalProperties": False,
        },
    }
]


# === Простые утилиты работы с профилем ===
def load_profile() -> Dict[str, Any]:
    if PROFILE_PATH.exists():
        try:
            with PROFILE_PATH.open("r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, dict) else {}
        except Exception:
            return {}
    return {}


def save_profile(profile: Dict[str, Any]) -> None:
    PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with PROFILE_PATH.open("w", encoding="utf-8") as f:
        json.dump(profile, f, ensure_ascii=False, indent=2)


def apply_profile_updates(updates: Dict[str, Any]) -> Dict[str, Any]:
    """
    Фильтруем и применяем ТОЛЬКО валидные и реально изменившиеся значения.
    Разрешённые ключи: sex(m|f), age(int), weight_kg(float)
    """
    allowed = {"sex", "age", "weight_kg"}
    updates = {k: v for k, v in (updates or {}).items() if k in allowed}
    if not updates:
        return {}

    current = load_profile()
    applied: Dict[str, Any] = {}

    # sex
    if "sex" in updates and updates["sex"] is not None:
        sex = str(updates["sex"]).lower()
        if sex in {"m", "f"} and current.get("sex") != sex:
            applied["sex"] = sex

    # age
    if "age" in updates and updates["age"] is not None:
        try:
            age_val = int(updates["age"])  # приведём к int
            if age_val > 0 and current.get("age") != age_val:
                applied["age"] = age_val
        except Exception:
            pass

    # weight_kg
    if "weight_kg" in updates and updates["weight_kg"] is not None:
        try:
            w = float(updates["weight_kg"])  # приведём к float
            # если в профиле было 80, а приходит 80.0 — считаем равным
            cw = current.get("weight_kg")
            if not (isinstance(cw, (int, float)) and abs(float(cw) - w) < 1e-6):
                if w > 0:
                    applied["weight_kg"] = w
        except Exception:
            pass

    if not applied:
        return {}

    current.update(applied)
    save_profile(current)
    return applied


def extract_and_update_profile(user_text: str) -> Dict[str, Any]:
    """
    1) Просим модель извлечь явно названные поля профиля через tools
    2) Если модель вернула вызов функции — применяем изменения к JSON
    """
    if not user_text:
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
        resp = client.responses.create(
            model=MODEL,
            instructions=instructions,
            input=user_text,
            tools=PROFILE_SAVE_TOOL,
            tool_choice="auto",
        )
    except Exception as exc:
        print(f"[tools] Ошибка вызова модели: {exc}")
        return {}

    # Ищем function_call save_profile_fields
    for item in getattr(resp, "output", []) or []:
        if getattr(item, "type", None) in ("function_call", "tool_call") and getattr(item, "name", "") == "save_profile_fields":
            args_raw = getattr(item, "arguments", "") or getattr(item, "input", "")
            try:
                args = json.loads(args_raw) if isinstance(args_raw, str) else (args_raw or {})
            except Exception:
                args = {}
            if isinstance(args, dict) and args:
                return apply_profile_updates(args)
            break
    return {}


def format_profile_for_prompt(profile: Dict[str, Any]) -> str:
    if not profile:
        return ""
    parts = []
    if profile.get("sex") in {"m", "f"}:
        parts.append("пол мужской" if profile["sex"] == "m" else "пол женский")
    if isinstance(profile.get("age"), int) and profile["age"] > 0:
        parts.append(f"возраст {profile['age']}")
    if isinstance(profile.get("weight_kg"), (int, float)) and float(profile["weight_kg"]) > 0:
        parts.append(f"вес {float(profile['weight_kg']):.2f} кг")
    return ("Профиль: " + ", ".join(parts)) if parts else ""


def generate_assistant_reply(user_text: str) -> str:
    """Простой короткий ответ (2–4 предложения) с учётом профиля (если есть)."""
    profile = load_profile()
    profile_ctx = format_profile_for_prompt(profile)
    system_prompt = "Ты — дружелюбный и краткий консультант по питанию/здоровью. Отвечай 2–4 предложениями."
    instructions = system_prompt if not profile_ctx else f"{system_prompt}\n\n{profile_ctx}"

    try:
        resp = client.responses.create(
            model=MODEL,
            instructions=instructions,
            input=user_text or "",
            max_output_tokens=500,
        )
    except Exception as exc:
        return f"[assistant] Ошибка вызова модели: {exc}"

    text = getattr(resp, "output_text", None)
    if isinstance(text, str) and text:
        return text

    # Резервный сбор текста
    chunks = []
    for item in getattr(resp, "output", []) or []:
        for c in getattr(item, "content", None) or []:
            t = getattr(c, "text", None)
            if isinstance(t, str) and t:
                chunks.append(t)
    return "\n\n".join(chunks) if chunks else "[assistant] Пустой ответ"


def main() -> None:
    print("Простой мини‑чат (Function Calling). /exit для выхода.")
    print("Профиль:", PROFILE_PATH)

    while True:
        try:
            user_text = input("\nВы: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nВыход.")
            break

        if not user_text or user_text.lower() == "/exit":
            print("Выход.")
            break

        # 1) Попробуем обновить профиль из текущего сообщения
        applied = extract_and_update_profile(user_text)
        if applied:
            print("Ассистент: ✅ Профиль обновлён:")
            print(json.dumps(applied, ensure_ascii=False))

        # 2) Короткий ответ ассистента
        reply = generate_assistant_reply(user_text)
        print("Ассистент:")
        print(reply)


if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("Требуется переменная окружения OPENAI_API_KEY")
    else:
        main()


