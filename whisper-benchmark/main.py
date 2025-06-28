import ctypes
import json
import logging
import os

from app import Benchmark, BenchmarkConfig

logging.basicConfig(
    level="INFO",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("whisper-benchmark.log", encoding="utf-8"),
    ],
)
logging.getLogger("faster_whisper").setLevel(logging.WARNING)
logging.getLogger("speechbrain.utils.fetching").setLevel(logging.WARNING)
logging.getLogger("speechbrain.utils.checkpoints").setLevel(logging.WARNING)
logging.getLogger("speechbrain.utils.parameter_transfer").setLevel(logging.WARNING)
logging.getLogger("pytorch_lightning.utilities.upgrade_checkpoint").setLevel(logging.WARNING)


def prevent_sleep():
    if os.name == "nt":
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000002)


def restore_sleep():
    if os.name == "nt":
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


def main():
    logger = logging.getLogger("whisper-benchmark")
    logger.info("Запуск бенчмарка whisper-consumer")

    prevent_sleep()
    try:
        config = BenchmarkConfig(**json.load(open("config.json", "r", encoding="utf-8")))
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
