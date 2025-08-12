import os
from dotenv import load_dotenv
import logging
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
ENABLE_DIALOG_LOGGING = True

# Часовой пояс по умолчанию для напоминаний
TIMEZONE = os.getenv("TIMEZONE", "Asia/Novosibirsk")
LOCAL_TZ = pytz.timezone(TIMEZONE)

# Настройка логгера
logging.basicConfig(
    level=getattr(logging, LOGGING_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Параметры запросов к OpenAI
OPENAI_PARAMS = {
    "temperature": 0.7,
    "max_tokens": 1000,
} 

# Список разрешенных пользователей (Telegram User ID)
ALLOWED_USERS = [
    1710039052,    # Ваш ID
    # 987654321,    # ID второго пользователя (закомментировано)
    # 111222333,  # ID третьего пользователя (закомментировано)
]