import os
from dotenv import load_dotenv
import logging
from logging.handlers import TimedRotatingFileHandler
import pytz

# Загрузка переменных окружения из файла .env
load_dotenv()

# Получение токенов и настроек из переменных окружения
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = "gpt-5-mini"
# Модель для транскрибации голосовых сообщений
TRANSCRIPTION_MODEL = "gpt-4o-mini-transcribe"

# Настройки логирования
LOGGING_LEVEL = "INFO"

# Управление историей диалога (памятью) в БД:
# True — сохраняем сообщения user/assistant в SQLite и
# используем их как контекст при обращении к модели.
# False — история не накапливается, контекст для модели минимальный.
ENABLE_MESSAGE_HISTORY = True

# Часовые пояса
# По умолчанию — Новосибирск; дополнительно поддерживаем Москву для выбранных пользователей
TIMEZONE = os.getenv("TIMEZONE", "Asia/Novosibirsk")
LOCAL_TZ = pytz.timezone(TIMEZONE)
MOSCOW_TZ = pytz.timezone("Europe/Moscow")

# Настройка логгера: ротация файлов и вывод в консоль
LOG_LEVEL = getattr(logging, LOGGING_LEVEL, logging.INFO)
LOG_DIR = os.path.join("data", "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "bot.log")

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

root_logger = logging.getLogger()
root_logger.setLevel(LOG_LEVEL)

# Избегаем дублирующих хендлеров при повторном импортировании backupCount= это количество сохраненых файлов 0 значит без ограничений
if not root_logger.handlers:
    file_handler = TimedRotatingFileHandler(
        LOG_FILE, when="midnight", backupCount=90, encoding="utf-8"
    )
    file_handler.setLevel(LOG_LEVEL)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(LOG_LEVEL)
    console_handler.setFormatter(formatter)

    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

logger = logging.getLogger("vibe_checker")

# Параметры запросов к OpenAI
OPENAI_PARAMS = {
    "temperature": 0.7, # не применяется в Responses API для GPT-5 mini
    "max_tokens": 5000,
}

# Включать ли краткий контекст задач/напоминаний в промпт модели
INCLUDE_REMINDERS_IN_PROMPT = True
# Включать ли краткий профиль пользователя (пол, возраст, рост, вес, активность, цель и т.д.) в промпт модели
INCLUDE_PROFILE_IN_PROMPT = True

# Список разрешенных пользователей (Telegram User ID)
ALLOWED_USERS = [
    1710039052,    # Ваш ID
    847720042,    # ID Татьяна Прокофьева
    1478674489,  # ID Таня
    484231158,  # ID Дарья
    7226852921,  # ID М
]

# Пользователи, для которых использовать московский часовой пояс
MOSCOW_USERS = [
    847720042,    # ID Татьяна Прокофьева
]

def get_user_tz(user_id: int):
    """
    Возвращает объект часового пояса для пользователя.
    По умолчанию — LOCAL_TZ (Новосибирск), для MOSCOW_USERS — московское время.
    """
    return MOSCOW_TZ if int(user_id) in MOSCOW_USERS else LOCAL_TZ