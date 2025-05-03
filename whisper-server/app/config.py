import os
import json
import logging
from typing import Dict, Any

logger = logging.getLogger("whisper-server")


def load_config() -> Dict[str, Any]:
    """
    Загружает конфигурацию из файла config.json и переменных окружения.
    Переменные окружения имеют приоритет над значениями из файла.

    Returns:
        словарь с конфигурацией
    """
    config_path = os.getenv("CONFIG_PATH", "config.json")
    config = {}

    try:
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
                logger.info(f"Конфигурация загружена из {config_path}")
        else:
            logger.warning(f"Файл конфигурации {config_path} не найден, используются значения по умолчанию")
    except Exception as e:
        logger.error(f"Ошибка при чтении конфигурации из {config_path}: {str(e)}")

    apply_env_variables(config)

    return config


def apply_env_variables(config: Dict[str, Any]) -> None:
    """
    Применяет переменные окружения к конфигурации.

    Args:
        config: словарь с конфигурацией
    """
    if "whisper" not in config:
        config["whisper"] = {}

    model_name = os.getenv("WHISPER_MODEL")
    if model_name:
        config["whisper"]["model_name"] = model_name
        logger.info(f"Используется модель из переменных окружения: {model_name}")

    device = os.getenv("WHISPER_DEVICE")
    if device:
        config["whisper"]["device"] = device

    compute_type = os.getenv("WHISPER_COMPUTE_TYPE")
    if compute_type and device:
        if isinstance(config["whisper"].get("compute_type", {}), dict):
            if device == "cuda" or "cuda" in device:
                config["whisper"]["compute_type"]["cuda"] = compute_type
            else:
                config["whisper"]["compute_type"]["cpu"] = compute_type
        else:
            config["whisper"]["compute_type"] = compute_type

    beam_size = os.getenv("WHISPER_BEAM_SIZE")
    if beam_size:
        config["whisper"]["beam_size"] = int(beam_size)

    best_of = os.getenv("WHISPER_BEST_OF")
    if best_of:
        config["whisper"]["best_of"] = int(best_of)

    download_root = os.getenv("WHISPER_DOWNLOAD_ROOT")
    if download_root:
        config["whisper"]["download_root"] = download_root


def get_whisper_config() -> Dict[str, Any]:
    """
    Возвращает конфигурацию для модели Whisper.

    Returns:
        словарь с конфигурацией модели
    """
    config = load_config()
    return config.get("whisper", {})
