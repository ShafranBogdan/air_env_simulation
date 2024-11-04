import logging

class Logger:
    def __init__(self, name: str, level: int = logging.DEBUG, log_file: str = None):
        """
        Инициализация логгера.
        
        :param name: Имя логгера, например, имя модуля.
        :param level: Уровень логирования. По умолчанию DEBUG.
        :param log_file: Путь к файлу для сохранения логов. Если None, логи выводятся только в консоль.
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # Устанавливаем формат логов
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        
        # Если указан лог-файл, добавляем файл в качестве обработчика
        if log_file:
            file_handler = logging.FileHandler(log_file, mode='w')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        else:
            # Создаем и добавляем обработчик для консоли
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

    def debug(self, message: str):
        self.logger.debug(message)

    def info(self, message: str):
        self.logger.info(message)

    def warning(self, message: str):
        self.logger.warning(message)

    def error(self, message: str):
        self.logger.error(message)

    def critical(self, message: str):
        self.logger.critical(message)
