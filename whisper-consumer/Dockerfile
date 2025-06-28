FROM python:3.12-slim

RUN pip install uv

WORKDIR /app

# Копируем whisper-model пакет
COPY whisper-model /whisper-model
RUN cd /whisper-model && uv pip install --system .

COPY whisper-consumer/pyproject.toml ./

RUN uv pip install --system .

COPY whisper-consumer/ .

ENV PYTHONUNBUFFERED=1

CMD ["dramatiq", "main"] 