import os

RABBITMQ_URL = os.getenv("RABBITMQ_URL", "amqp://guest:guest@localhost:5672/")
TASK_QUEUE_NAME = os.getenv("TASK_QUEUE_NAME")
RESULTS_QUEUE_NAME = os.getenv("RESULTS_QUEUE_NAME")
WHISPER_CONFIG_JSON_PATH = os.getenv("WHISPER_CONFIG_JSON_PATH")

BATCH_ACCUMULATION_TIME_S = int(os.getenv("BATCH_ACCUMULATION_TIME_S", "45"))
BATCH_MAX_TOTAL_DURATION_S = int(os.getenv("BATCH_MAX_TOTAL_DURATION_S", "1800"))
BATCH_MAX_FILES = int(os.getenv("BATCH_MAX_FILES", "6"))
SINGLE_FILE_MAX_DURATION_S = 60

if not TASK_QUEUE_NAME or not RESULTS_QUEUE_NAME:
    raise ValueError(
        f"Переменные окружения TASK_QUEUE_NAME и RESULTS_QUEUE_NAME должны быть установлены."
        f" Текущие значения: {TASK_QUEUE_NAME=}, {RESULTS_QUEUE_NAME=}"
    )

if not WHISPER_CONFIG_JSON_PATH:
    raise ValueError("Переменная окружения WHISPER_CONFIG_JSON_PATH должна быть установлена и указывать на JSON файл конфигурации.")
