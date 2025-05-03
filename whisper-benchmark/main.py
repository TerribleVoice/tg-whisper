import argparse
import logging
import os

from app import Benchmark, Config


logging.basicConfig(
    level="INFO",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("whisper-benchmark.log", encoding="utf-8")],
)
logging.getLogger("faster_whisper").setLevel(logging.WARNING)


def prevent_sleep():
    if os.name == "nt":
        import ctypes

        ctypes.windll.kernel32.SetThreadExecutionState(0x80000002)


def restore_sleep():
    if os.name == "nt":
        import ctypes

        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


def parse_args():
    parser = argparse.ArgumentParser(description="Бенчмарк для whisper-server")
    parser.add_argument("--config", type=str, default="config.json", help="Путь к файлу конфигурации (по умолчанию: config.json)")
    return parser.parse_args()


def main():
    args = parse_args()

    logger = logging.getLogger("whisper-benchmark")
    logger.info("Запуск бенчмарка whisper-server")
    logger.info(f"Файл конфигурации: {args.config}")

    prevent_sleep()
    try:
        config = Config(args.config)
        benchmark = Benchmark(config)
        benchmark.run()

        logger.info("Бенчмарк успешно завершен")
    except Exception as e:
        logger.error(f"Ошибка при выполнении бенчмарка: {str(e)}", exc_info=True)
        return 1
    finally:
        restore_sleep()

    return 0


if __name__ == "__main__":
    exit(main())
