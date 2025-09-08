## Короткое руководство: Function Calling в OpenAI (для новичка)

Цель: понять суть и быстро запустить простой пример (см. `examples/minimal_ai_consultant_simple.py`).

### 1) Идея
- Вы описываете «инструменты» (`tools`) — по сути, функции с именем, описанием и схемой параметров (JSON Schema).
- Модель сама решает, когда вызвать функцию, и присылает вам `function_call` с аргументами.
- Ваша программа принимает эти аргументы, вызывает свою реальную функцию/логику (сохранение в БД, HTTP‑запрос и т. п.), а затем продолжает диалог.

В этом репозитории мы делаем простую вещь: извлекаем поля профиля (sex, age, weight_kg) и сохраняем их в `profile.json`.

### 2) Как описать инструмент
В параметре `tools` передаём список. Каждый инструмент — объект:

```json
{
  "type": "function",
  "name": "save_profile_fields",
  "description": "Коротко: что делает инструмент",
  "strict": true,
  "parameters": {
    "type": "object",
    "properties": {
      "sex": { "type": ["string", "null"], "enum": ["m", "f"] },
      "age": { "type": ["integer", "null"] },
      "weight_kg": { "type": ["number", "null"] }
    },
    "required": ["sex", "age", "weight_kg"],
    "additionalProperties": false
  }
}
```

Главное: схема параметров. Модель старается следовать ей и не придумывать поля вне схемы.

### 3) Как сделать запрос с tools
Пример (на Python, через OpenAI SDK) — см. функцию `extract_and_update_profile` в `minimal_ai_consultant_simple.py`:

```python
resp = client.responses.create(
    model=MODEL,
    instructions="Ты — парсер профиля...",
    input=user_text,
    tools=PROFILE_SAVE_TOOL,
    tool_choice="auto",
)
```

- `tool_choice="auto"` — модель сама решает, звать ли функцию.
- Дальше нужно проверить `resp.output` и найти элемент с `type == "function_call"` и `name == "save_profile_fields"`.
- Аргументы лежат в `item.arguments` (строка JSON) — распарсить, затем применить.

### 4) Что делать с результатами функции
Есть два популярных подхода:
- Простой (как у нас): применили изменения локально (например, сохранили в `profile.json`) и сделали новый независимый вызов модели для ответа.
- «Tool loop»: после выполнения отправляем «вывод инструмента» обратно в ту же сессию ответа, чтобы модель продолжила шаг рассуждений. Это сложнее, но даёт более глубокую интеграцию. Для старта не требуется.

### 5) Мини‑план действий
1. Описать инструмент в `tools` (имя, описание, JSON Schema).
2. Сделать запрос `responses.create(..., tools=..., tool_choice="auto")`.
3. Если пришёл `function_call` — распарсить `arguments`, выполнить свою логику, сохранить результат.
4. Сформировать «системный» контекст (например, кратко из профиля) и сделать второй вызов модели для ответа пользователю.

### 6) Что читать дальше
- Официальные доки OpenAI по Function Calling: `https://platform.openai.com/docs/guides/function-calling`.
- Пример из этого репозитория: `examples/minimal_ai_consultant_simple.py` (самый короткий), `examples/minimal_ai_consultant.py` (чуть сложнее, с логами).

Этого достаточно, чтобы понять основы и начать экспериментировать.


