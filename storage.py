"""
Модуль для хранения и управления данными пользователей.
"""

import json
import os
import time
from typing import Dict, List, Any, Optional, Tuple
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
                "last_interaction": time.time(),
                # Список напоминаний пользователя
                # Формат элемента: {id, time: "HH:MM", text: str, last_sent_date: "YYYY-MM-DD" | None}
                "reminders": [],
                # Последний известный chat_id для отправки системных сообщений
                "chat_id": None,
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

    # ===== Напоминания =====
    def set_chat_id(self, user_id: int, chat_id: int) -> None:
        """
        Сохраняет последний известный chat_id пользователя для отправки напоминаний.
        """
        user_data = self.get_user_data(user_id)
        user_data["chat_id"] = int(chat_id)
        self._save_data()

    def get_chat_id(self, user_id: int) -> Optional[int]:
        """
        Возвращает сохраненный chat_id пользователя, если есть.
        """
        user_data = self.get_user_data(user_id)
        return user_data.get("chat_id")

    def add_reminder(self, user_id: int, time_hh_mm: str, text: str) -> Dict[str, Any]:
        """
        Добавляет напоминание пользователю.

        time_hh_mm должен быть в формате HH:MM (24 часа).
        """
        # Валидация формата времени
        try:
            hours_str, minutes_str = time_hh_mm.split(":")
            hours = int(hours_str)
            minutes = int(minutes_str)
            if not (0 <= hours <= 23 and 0 <= minutes <= 59):
                raise ValueError
        except Exception:
            raise ValueError("Неверный формат времени. Используйте HH:MM, например 09:30")

        user_data = self.get_user_data(user_id)
        reminders: List[Dict[str, Any]] = user_data.setdefault("reminders", [])

        # Генерация нового ID
        new_id = 1
        if reminders:
            try:
                new_id = max(int(r.get("id", 0)) for r in reminders) + 1
            except Exception:
                new_id = len(reminders) + 1

        reminder = {
            "id": new_id,
            "time": f"{hours:02d}:{minutes:02d}",
            "text": text.strip(),
            "last_sent_date": None,
        }
        reminders.append(reminder)
        self._save_data()
        return reminder

    def list_reminders(self, user_id: int) -> List[Dict[str, Any]]:
        """
        Возвращает список напоминаний пользователя.
        """
        user_data = self.get_user_data(user_id)
        return list(user_data.get("reminders", []))

    def delete_reminder(self, user_id: int, identifier: int) -> bool:
        """
        Удаляет напоминание по ID или по порядковому номеру в списке (1..N).

        Возвращает True, если удалено, иначе False.
        """
        user_data = self.get_user_data(user_id)
        reminders: List[Dict[str, Any]] = user_data.get("reminders", [])
        if not reminders:
            return False

        # Пытаемся как ID
        for idx, r in enumerate(reminders):
            try:
                if int(r.get("id")) == int(identifier):
                    del reminders[idx]
                    self._save_data()
                    return True
            except Exception:
                pass

        # Пытаемся как порядковый номер
        index_zero_based = int(identifier) - 1
        if 0 <= index_zero_based < len(reminders):
            del reminders[index_zero_based]
            self._save_data()
            return True
        return False

    def get_due_reminders(self, time_hh_mm: str, today_date: str) -> List[Tuple[int, Dict[str, Any]]]:
        """
        Возвращает список напоминаний, которые должны сработать в указанное время.

        Чтобы избежать повторной отправки в тот же день, используется поле last_sent_date.
        """
        due: List[Tuple[int, Dict[str, Any]]] = []
        for user_id_str, user_data in self.users_data.items():
            reminders: List[Dict[str, Any]] = user_data.get("reminders", [])
            for r in reminders:
                if r.get("time") == time_hh_mm and r.get("last_sent_date") != today_date:
                    try:
                        user_id_int = int(user_id_str)
                        due.append((user_id_int, r))
                    except Exception:
                        continue
        return due

    def mark_reminder_sent(self, user_id: int, reminder_id: int, today_date: str) -> None:
        """
        Помечает напоминание как отправленное сегодня, чтобы не дублировать в течение дня.
        """
        user_data = self.get_user_data(user_id)
        reminders: List[Dict[str, Any]] = user_data.get("reminders", [])
        for r in reminders:
            try:
                if int(r.get("id")) == int(reminder_id):
                    r["last_sent_date"] = today_date
                    break
            except Exception:
                continue
        self._save_data()
    
    def get_message_history(self, user_id: int, max_messages: int = 50) -> List[Dict[str, str]]:
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