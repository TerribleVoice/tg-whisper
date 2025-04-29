from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from pydantic import BaseModel
from typing import Optional
import logging

from .service import TranscriptionService

logger = logging.getLogger("whisper-server")

app = FastAPI(title="Whisper Transcription API")

transcription_service = TranscriptionService(model_name="tiny")


class TranscriptionResponse(BaseModel):
    text: str


@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(audio_file: UploadFile = File(...), language: Optional[str] = Form(None)):
    """
    Транскрибировать аудиофайл.

    Args:
        audio_file: Аудиофайл для транскрипции
        language: Язык аудио (опционально)

    Returns:
        Результат транскрипции в виде текста
    """
    logger.info(f"Получен запрос на транскрипцию файла: {audio_file.filename}")

    if not audio_file.filename or not any(audio_file.filename.endswith(ext) for ext in (".mp3", ".wav", ".ogg", ".flac", ".m4a")):
        logger.error(f"Файл с неподдерживаемым форматом: {audio_file.filename}")
        raise HTTPException(status_code=400, detail="Неподдерживаемый формат аудио. Поддерживаемые форматы: mp3, wav, ogg, flac, m4a")

    audio_data = await audio_file.read()
    if not audio_data:
        logger.error("Получен пустой аудиофайл")
        raise HTTPException(status_code=400, detail="Пустой аудиофайл")

    try:
        text = await transcription_service.transcribe_audio(audio_data, language)
        logger.info("Транскрипция успешно завершена")
        return TranscriptionResponse(text=text)
    except Exception as e:
        logger.error(f"Ошибка при транскрипции: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ошибка транскрипции: {str(e)}")


@app.get("/health")
async def health_check():
    """
    Проверка состояния сервера.

    Returns:
        Статус сервера
    """
    logger.debug("Получен запрос на проверку состояния сервера")
    return {"status": "ok"}
