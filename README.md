# Бот-помощник по делам

Телеграм-бот, используя модель gpt-4.1-mini от OpenAI.

- Сохранение истории диалогов
- Естественный диалог с уточнениями

## Технологии

- Python 3.10+
- Aiogram 3.19.0 (Telegram Bot API)
- OpenAI API 1.66.3 (модель gpt-4.1-mini)
- Асинхронное программирование

## Установка и запуск

1. Клонируйте репозиторий:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Создайте виртуальное окружение и активируйте его:
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. Установите зависимости:
```bash
pip install -r requirements.txt
```

4. Создайте файл `.env` в корневой директории проекта со следующим содержимым:
```
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-4.1-mini
LOG_LEVEL=INFO
ENABLE_DIALOG_LOGGING=true
```

5. Запустите бота:
```bash
python main.py
```

## Команды бота

- `/start` - Начать диалог с ботом
- `/help` - Получить справку о возможностях бота
- `/clear` - Очистить историю диалога

## Структура проекта

- `main.py` - Основной файл бота
- `config.py` - Конфигурация и настройки
- `openai_module.py` - Логика взаимодействия с OpenAI API
- `storage.py` - Логика хранения данных пользователей
- `prompts.py` - Шаблоны промптов для OpenAI
- `requirements.txt` - Зависимости проекта 