# Telegram Bot Backend

Telegram бот для приема голосовых сообщений и постановки задач в очередь.

## Функциональность

- Прием голосовых сообщений от пользователей
- Получение URL файла через Telegram Bot API
- Постановка задач транскрипции в RabbitMQ очередь
- Поддержка webhook и polling режимов

## Технологии

- **aiogram** - асинхронная библиотека для Telegram Bot API
- **dramatiq** - система очередей задач
- **aiohttp** - веб-сервер для webhook режима

## Конфигурация

### Переменные окружения

- `BOT_TOKEN` - токен Telegram бота (обязательно)
- `RABBITMQ_URL` - URL подключения к RabbitMQ (по умолчанию: `amqp://guest:guest@localhost:5672/`)
- `TASK_QUEUE_NAME` - имя очереди для задач (обязательно)
- `TG_BOT_BACKEND_WEBHOOK_PATH` - путь для webhook (по умолчанию: `/webhook`)
- `TG_BOT_BACKEND_PORT` - порт для webhook режима (по умолчанию: 8080)
- `TG_BOT_BACKEND_USE_POLLING` - использовать polling вместо webhook (по умолчанию: `true`)
- `WEBHOOK_URL` - базовый URL для webhook
- `LOG_LEVEL` - уровень логирования (по умолчанию: `INFO`)

## Режимы работы

### Polling режим (по умолчанию)
```bash
TG_BOT_BACKEND_USE_POLLING=true
```
Бот опрашивает Telegram API для получения новых сообщений.

### Webhook режим
```bash
TG_BOT_BACKEND_USE_POLLING=false
WEBHOOK_URL=https://your-domain.com
```
Telegram отправляет сообщения на указанный webhook URL.

## Обработка сообщений

1. Пользователь отправляет голосовое сообщение
2. Бот получает `file_id` и запрашивает `file_path` через API
3. Формируется задача с параметрами:
   - `file_url` - URL для скачивания файла
   - `chat_id` - ID чата для отправки результата
   - `message_date` - дата сообщения в ISO формате
4. Задача помещается в RabbitMQ очередь
5. Пользователю отправляется подтверждение

## Особенности

- Обрабатываются только голосовые сообщения (`voice`)
- Текстовые сообщения просто эхо-ответ (для тестирования)
- Используется `dramatiq.Message` для прямой работы с очередью
- Корректная обработка lifecycle при запуске/остановке 