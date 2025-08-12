import os
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI


# --- НАСТРОЙКИ ТЕСТА (меняйте под себя) ---
# Модель для теста
MODEL: str = "gpt-5-mini"  # укажите актуальную модель для вашего тарифа

# Системная инструкция и сообщение пользователя
INSTRUCTIONS: str = "Ты ИИ ассистент, отвечай лаконично."
USER_MESSAGE: str = "Скажи нужную при разработке на Python фишку OpenAI API client.responses.create."

# Лимит генерируемых токенов и температура (опционально)
# Поставьте None, чтобы не задавать параметр
MAX_OUTPUT_TOKENS: Optional[int] = 2000
TEMPERATURE: Optional[float] = 0.7  # нельзя выбрать в gpt-5


def get_required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Переменная окружения {name} не задана в .env")
    return value


def ask_model_with_responses(
    user_message: str,
    *,
    model: str,
    instructions: Optional[str] = None,
    max_output_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
) -> str:
    """Отправляет текстовый запрос через OpenAI Responses API и возвращает ответ.

    Важно: токен берём из окружения (OPENAI_API_KEY), все остальные параметры — из аргументов.
    """

    api_key = get_required_env("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)

    kwargs = {
        "model": model,
        "input": user_message,
    }
    if instructions:
        kwargs["instructions"] = instructions
    if isinstance(max_output_tokens, int) and max_output_tokens > 0:
        kwargs["max_output_tokens"] = max_output_tokens
    temperature_used: bool = False
    if isinstance(temperature, (int, float)):
        kwargs["temperature"] = float(temperature)
        temperature_used = True

    # Пытаемся вызвать API; если модель не поддерживает temperature — пробуем без него
    try:
        response = client.responses.create(**kwargs)
    except Exception as exc:
        msg = str(exc)
        if "Unsupported parameter" in msg and "'temperature'" in msg:
            kwargs.pop("temperature", None)
            temperature_used = False
            response = client.responses.create(**kwargs)
        else:
            raise

    # --- Читаемый лог об использовании токенов ---
    def _safe_get(obj, name):
        try:
            return getattr(obj, name)
        except Exception:
            return None

    usage = _safe_get(response, "usage")
    input_tokens = None
    output_tokens = None
    total_tokens = None
    cached_tokens = None
    reasoning_tokens = None

    if usage is not None:
        # Пытаемся извлечь основные поля разными путями для совместимости версий SDK
        total_tokens = (
            _safe_get(usage, "total_tokens")
            or _safe_get(usage, "tot_tokens")
        )
        input_tokens = (
            _safe_get(usage, "input_tokens")
            or _safe_get(usage, "prompt_tokens")
            or _safe_get(usage, "input_text_tokens")
        )
        output_tokens = (
            _safe_get(usage, "output_tokens")
            or _safe_get(usage, "completion_tokens")
            or _safe_get(usage, "output_text_tokens")
        )

        input_details = _safe_get(usage, "input_tokens_details")
        if input_details is not None:
            cached_tokens = _safe_get(input_details, "cached_tokens")

        output_details = _safe_get(usage, "output_tokens_details")
        if output_details is not None:
            # reasoning_tokens может лежать в деталях выхода
            reasoning_tokens = (
                _safe_get(output_details, "reasoning_tokens")
                or _safe_get(usage, "reasoning_tokens")
            )

    print("Информация об использовании токенов:")
    print(f"- Модель: {model}")
    print(
        f"- max_output_tokens: {max_output_tokens if isinstance(max_output_tokens, int) and max_output_tokens > 0 else 'не задано'}"
    )
    print(f"- temperature: {'применена' if temperature_used else 'не применена'}")

    if any(v is not None for v in (input_tokens, output_tokens, total_tokens, cached_tokens, reasoning_tokens)):
        parts = []
        if input_tokens is not None:
            parts.append(f"input={input_tokens}")
        if output_tokens is not None:
            parts.append(f"output={output_tokens}")
        if total_tokens is not None:
            parts.append(f"total={total_tokens}")
        print("- Токены: " + ", ".join(parts) if parts else "- Токены: недоступно")

        details_parts = []
        if cached_tokens is not None:
            details_parts.append(f"cached={cached_tokens}")
        if reasoning_tokens is not None:
            details_parts.append(f"reasoning={reasoning_tokens}")
        if details_parts:
            print("- Детали: " + ", ".join(details_parts))
    else:
        print("- Токены: недоступно в ответе SDK этой версии")

    # Прямое извлечение текста (упрощённо для текущего SDK)
    text = response.output_text
    return text.strip() if text else str(response)


def main() -> None:
    load_dotenv(override=True)

    try:
        answer = ask_model_with_responses(
            USER_MESSAGE,
            model=MODEL,
            instructions=INSTRUCTIONS,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            temperature=TEMPERATURE,
        )
        print("Ответ модели:")
        print(answer)
    except Exception as exc:
        print("❌ Ошибка при обращении к OpenAI Responses API:")
        print(str(exc))
        print("Проверьте OPENAI_API_KEY в .env, интернет и корректность параметров модели.")


if __name__ == "__main__":
    main()


