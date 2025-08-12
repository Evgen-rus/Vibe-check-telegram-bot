На август 2025 API-метод ChatCompletion больше не поддерживается в последних версиях OpenAI API и официальных клиентах. Это связано с переходом OpenAI на новые эндпоинты и модели, особенно после релиза GPT-5 и деактивации ряда старых моделей, включая o3, o4, GPT-4o и др.

# Инструкция по миграции с старого метода `client.chat.completions.create` на новый `client.responses.create` (Responses API). 

### Что меняется по сути
- Было: `chat.completions` с полем `messages` и параметром `max_tokens`.
- Стало: `responses.create` с полями `instructions` + `input` и параметром `max_output_tokens`.

### Минимальная замена (до/после)
- Было:
  ```python
  from openai import OpenAI
  client = OpenAI(api_key=OPENAI_API_KEY)

  resp = client.chat.completions.create(
      model=MODEL,
      messages=[
          {"role": "system", "content": "Системная инструкция"},
          {"role": "user", "content": user_message},
      ],
      max_tokens=200,
  )
  text = resp.choices[0].message.content.strip()
  ```
- Стало:
  ```python
  from openai import OpenAI
  client = OpenAI(api_key=OPENAI_API_KEY)

  resp = client.responses.create(
      model=MODEL,
      instructions="Системная инструкция",
      input=user_message,
      # опционально:
      # max_output_tokens=200,
  )
  text = resp.output_text.strip()
  ```

### Карта замен параметров
- max_tokens → max_output_tokens
- messages (system+user) → instructions + input
- Доступ к тексту:
  - раньше: `resp.choices[0].message.content`
  - теперь: `resp.output_text` (или собрать из `resp.output` при необходимости)

### Типовая функция-обёртка
```python
def get_ai_answer(user_message: str) -> str:
    system_prompt = "Твоя системная инструкция"
    resp = client.responses.create(
        model=MODEL,
        instructions=system_prompt,
        input=user_message,
        # max_output_tokens=200,  # если хотите ограничить длину
    )
    return (resp.output_text or "").strip()
```

### Как получить текст из ответа
- Удобно: resp.output_text (если доступно).
- Надёжно: resp.output[0].content[0].text

Пример (синхронно):
from openai import OpenAI
client = OpenAI()

resp = client.responses.create(
    model="gpt-4o-mini",
    input="Напиши короткий план статьи про тестирование кода",
    temperature=0.2,
    max_output_tokens=300
)
print(resp.output_text or resp.output[0].content[0].text)


### Частые ошибки и решения
- 400 unsupported parameter 'max_tokens' → замените на `max_output_tokens`.
- Пустой текст в ответе → берите `resp.output_text`; если None, соберите текст из `resp.output` по элементам типа `output_text`.
- Старый импорт `openai.ChatCompletion` → используйте только `from openai import OpenAI` и `client = OpenAI(...)`.

### Проверка после миграции
- Отправьте простой запрос и убедитесь, что приходит ответ без ошибок.
- При необходимости добавьте в `.env` переменную `OPENAI_MAX_OUTPUT_TOKENS` и прокиньте её в `max_output_tokens`.

IMPLEMENTATION CHECKLIST:
1. Обновить вызовы: `client.chat.completions.create(...)` → `client.responses.create(...)`.
2. Перенести system‑prompt в `instructions`, текст пользователя в `input`.
3. Заменить `max_tokens` на `max_output_tokens` (или убрать ограничение).
4. Доставать текст через `response.output_text`.
5. Прогнать минимальный тест запроса и убедиться в отсутствии 400‑ошибок.

просто для справки
Ответ модели:
Response(id='resp_6899d0fe730c819585a9c7d7d80a64e70afa9d87cbadb470', created_at=1754910974.0, error=None, incomplete_details=IncompleteDetails(reason='max_output_tokens'), instructions='Ты ИИ ассистент, отвечай лаконично.', metadata={}, model='gpt-5-mini-2025-08-07', object='response', output=[ResponseReasoningItem(id='rs_6899d0fefc0c8195817d073c7c2451830afa9d87cbadb470', summary=[], type='reasoning', encrypted_content=None, status=None)], parallel_tool_calls=True, temperature=1.0, tool_choice='auto', tools=[], top_p=1.0, background=False, max_output_tokens=16, max_tool_calls=None, previous_response_id=None, prompt=None, prompt_cache_key=None, reasoning=Reasoning(effort='medium', generate_summary=None, summary=None), safety_identifier=None, service_tier='default', status='incomplete', text=ResponseTextConfig(format=ResponseFormatText(type='text'), verbosity='medium'), top_logprobs=0, truncation='disabled', usage=ResponseUsage(input_tokens=32, input_tokens_details=InputTokensDetails(cached_tokens=0), output_tokens=0, output_tokens_details=OutputTokensDetails(reasoning_tokens=0), total_tokens=32), user=None, store=True)