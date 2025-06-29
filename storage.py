"""
Модуль для хранения и управления данными пользователей.
"""

import json
import os
import time
from typing import Dict, List, Any, Optional
from config import logger, ENABLE_DIALOG_LOGGING

# Директория для хранения данных
DATA_DIR = "data"
USERS_FILE = os.path.join(DATA_DIR, "users.json")

# Создаем директорию для данных, если она не существует
os.makedirs(DATA_DIR, exist_ok=True)


class Storage:
    """
    Класс для хранения и управления данными пользователей.
    """
    def __init__(self):
        """
        Инициализация хранилища данных.
        Загружает существующие данные пользователей, если они есть.
        """
        self.users_data = {}
        self._load_data()
    
    def _load_data(self) -> None:
        """
        Загружает данные пользователей из файла.
        """
        if os.path.exists(USERS_FILE):
            try:
                with open(USERS_FILE, 'r', encoding='utf-8') as file:
                    self.users_data = json.load(file)
                logger.info(f"Данные пользователей загружены из {USERS_FILE}")
            except Exception as e:
                logger.error(f"Ошибка при загрузке данных пользователей: {e}")
                self.users_data = {}
    
    def _save_data(self) -> None:
        """
        Сохраняет данные пользователей в файл.
        """
        try:
            with open(USERS_FILE, 'w', encoding='utf-8') as file:
                json.dump(self.users_data, file, ensure_ascii=False, indent=2)
            logger.debug(f"Данные пользователей сохранены в {USERS_FILE}")
        except Exception as e:
            logger.error(f"Ошибка при сохранении данных пользователей: {e}")
    
    def get_user_data(self, user_id: int) -> Dict[str, Any]:
        """
        Получает данные пользователя по его ID.
        Если пользователь не существует, создает новую запись.
        
        Args:
            user_id: ID пользователя в Telegram
            
        Returns:
            Dict с данными пользователя
        """
        user_id_str = str(user_id)
        
        if user_id_str not in self.users_data:
            # Создаем новую запись для пользователя
            self.users_data[user_id_str] = {
                "messages": [],
                "last_interaction": time.time()
            }
            self._save_data()
            
        return self.users_data[user_id_str]
    
    def add_message(self, user_id: int, role: str, content: str) -> None:
        """
        Добавляет сообщение в историю диалога пользователя.
        
        Args:
            user_id: ID пользователя в Telegram
            role: Роль отправителя ('user' или 'assistant')
            content: Содержимое сообщения
        """
        if not ENABLE_DIALOG_LOGGING:
            return
            
        user_data = self.get_user_data(user_id)
        user_data["messages"].append({
            "role": role,
            "content": content
        })
        user_data["last_interaction"] = time.time()
        self._save_data()
    
    def get_message_history(self, user_id: int, max_messages: int = 10) -> List[Dict[str, str]]:
        """
        Получает историю сообщений пользователя.
        
        Args:
            user_id: ID пользователя в Telegram
            max_messages: Максимальное количество сообщений для возврата
            
        Returns:
            Список словарей сообщений в формате OpenAI: [{"role": "...", "content": "..."}]
        """
        user_data = self.get_user_data(user_id)
        # Возвращаем последние max_messages сообщений
        return user_data["messages"][-max_messages:]
    

    def clear_history(self, user_id: int) -> None:
        """
        Очищает историю сообщений пользователя.
        
        Args:
            user_id: ID пользователя в Telegram
        """
        user_data = self.get_user_data(user_id)
        user_data["messages"] = []
        self._save_data()
        

# Создаем экземпляр хранилища
storage = Storage() 