# Telegram Bot Consumer

Воркер для получения результатов транскрипции и отправки их пользователям.

## Функциональность

- Получение результатов транскрипции из RabbitMQ очереди
- Отправка результатов пользователям через Telegram Bot API
- Обработка ошибок транскрипции

## Технологии

- **aiogram** - для отправки сообщений в Telegram
- **dramatiq** - для обработки очередей
- **AsyncIO middleware** - для асинхронной работы с Telegram API

## Конфигурация

### Переменные окружения

- `BOT_TOKEN` - токен Telegram бота (обязательно)
- `RESULTS_QUEUE_NAME` - имя очереди для результатов (обязательно)
- `RABBITMQ_URL` - URL подключения к RabbitMQ (по умолчанию: `amqp://guest:guest@localhost:5672/`)
- `LOG_LEVEL` - уровень логирования (по умолчанию: `INFO`)

## Обработка результатов

Actor `handle_transcription_result` обрабатывает сообщения с параметрами:
- `original_chat_id` - ID чата для отправки результата
- `transcript` - текст транскрипции
- `error` - сообщение об ошибке (опционально)

### Успешная транскрипция
```python
{
    "original_chat_id": 123456789,
    "transcript": "Привет, как дела?",
    "error": None
}
```

### Ошибка транскрипции
```python
{
    "original_chat_id": 123456789,
    "transcript": None,
    "error": "Ошибка загрузки файла: 404"
}
```

## Запуск

```bash
# Запуск воркера
dramatiq telegram-bot-consumer.main

# Или с указанием количества процессов
dramatiq telegram-bot-consumer.main --processes 4
```

## Особенности

- Используется `AsyncIO` middleware для асинхронной отправки сообщений
- Логирование всех получаемых результатов
- Обработка как успешных результатов, так и ошибок
- Queue name и actor name совпадают для упрощения 