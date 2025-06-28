FROM python:3.12-slim

RUN pip install uv

WORKDIR /app

COPY pyproject.toml* ./ 

RUN uv pip install --system . # --system для установки в системный Python в контейнере

COPY . .

ENV PYTHONUNBUFFERED=1
ENV WEBHOOK_PATH="/webhook"
ENV PORT=8080
ENV USE_POLLING="false"
ENV USE_STUB_BROKER="false"

CMD ["python", "main.py"]

# Команда для запуска воркера dramatiq (если в этом сервисе будут консьюмеры)
# Например, для обработки результатов от whisper-consumer
# CMD ["dramatiq", "main"] # Укажите правильный модуль 