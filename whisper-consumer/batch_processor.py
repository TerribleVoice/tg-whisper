import asyncio
import logging
from typing import Dict, List, Optional

from batch_task import BatchTask
from config import (
    BATCH_ACCUMULATION_TIME_S,
    BATCH_MAX_TOTAL_DURATION_S,
)
from transcription import send_result
from whisper_model import TranscriptionResult, WhisperXModel


class BatchProcessor:
    def __init__(self, whisper_model: WhisperXModel, broker):
        self.whisper_model = whisper_model
        self.broker = broker
        self.pending_tasks: List[BatchTask] = []
        self.batch_timer: Optional[asyncio.Task] = None
        self.lock = asyncio.Lock()
        self.batch_duration = 0.0

    async def add_task(self, task: BatchTask):
        async with self.lock:
            self.pending_tasks.append(task)
            self.batch_duration += task.audio_duration

            logging.info(
                f"Задача добавлена в батч. Размер батча: {len(self.pending_tasks)}, общая длительность: {self.batch_duration:.1f}s"
            )

            if self.batch_duration >= BATCH_MAX_TOTAL_DURATION_S:
                await self._process_current_batch()
            elif self.batch_timer is None:
                self.batch_timer = asyncio.create_task(self._batch_timer_task())

    async def _batch_timer_task(self):
        await asyncio.sleep(BATCH_ACCUMULATION_TIME_S)
        async with self.lock:
            if self.pending_tasks:
                await self._process_current_batch()

    async def _process_current_batch(self):
        if not self.pending_tasks:
            return

        if self.batch_timer:
            self.batch_timer.cancel()
            self.batch_timer = None

        tasks_to_process = self.pending_tasks.copy()
        self.pending_tasks.clear()
        self.batch_duration = 0.0

        logging.info(f"Начинаю обработку батча из {len(tasks_to_process)} задач")
        await self._process_batch_tasks(tasks_to_process)

    async def _process_batch_tasks(self, tasks: List[BatchTask]):
        audio_files = [task.file_path for task in tasks]
        batch_results: Dict[str, TranscriptionResult] = {}

        try:
            logging.info(f"Начинаю пакетную транскрипцию {len(audio_files)} файлов")
            batch_results = self.whisper_model.transcribe_batch(audio_files)

        except Exception as e:
            logging.exception(f"Ошибка при пакетной обработке: {e}")

        finally:
            await self._send_batch_results(tasks, batch_results)
            for task in tasks:
                task.file_path.unlink(missing_ok=True)

    async def _send_batch_results(self, tasks: List[BatchTask], batch_results: Dict[str, TranscriptionResult]):
        if not batch_results:
            return

        send_tasks = []

        for task in tasks:
            file_name = task.file_path.name
            result_text = batch_results[file_name].text
            send_tasks.append(send_result(self.broker, task.chat_id, result_text, None))
            logging.info(f"Результат подготовлен для отправки chat_id: {task.chat_id}")

        await asyncio.gather(*send_tasks)
        logging.info(f"Все результаты отправлены для {len(tasks)} задач")
