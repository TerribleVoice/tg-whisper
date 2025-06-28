# Whisper Model Library

Библиотека для транскрипции голосовых файлов с использованием WhisperX.

## Возможности

- **Транскрипция** - преобразование речи в текст
- **Выравнивание** - точное позиционирование слов по времени
- **Сегментация** - разделение по спикерам/сегментам
- **Batch обработка** - транскрипция нескольких файлов одновременно
- **Метрики** - замер времени выполнения операций

## Архитектура

### Основные классы

- `WhisperXModel` - основной класс для транскрипции
- `WhisperXConfig` - конфигурация модели
- `TranscriptionResult` - результат транскрипции
- `TranscriptionMetrics` - метрики производительности
- `TextFormatter` - форматирование результатов

### Конфигурация

```python
class WhisperXConfig:
    whisper_config: WhisperConfig      # Настройки Whisper
    align_config: AlignConfig          # Настройки выравнивания
    segmentation_config: SegmentationConfig  # Настройки сегментации
```

## Использование

### Инициализация
```python
from whisper_model import WhisperXModel
from whisper_model.config import WhisperXConfig

config = WhisperXConfig.from_json("config.json")
model = WhisperXModel(config)
```

### Транскрипция одного файла
```python
result = model.transcribe(Path("audio.wav"))
print(result.text)
print(f"Время: {result.metrics.transcribe_time:.2f}s")
```

### Batch транскрипция
```python
results = model.transcribe_batch([
    Path("audio1.wav"),
    Path("audio2.wav")
])
```

## Особенности реализации

### Сегментация спикеров
- Использует **pyannote.audio** для определения смены спикеров
- Настраиваемые параметры через `PeakConfig`
- Слова группируются по сегментам

### Форматирование текста
- Первое слово каждого сегмента с заглавной буквы
- Сегменты разделяются символом "– " (тире с пробелом)
- Удаление пустых сегментов

### Batch обработка
- Склеивание файлов с паузами между ними
- Декомпозиция результатов обратно по файлам
- Пропорциональное распределение метрик

### Производительность
- Подавление stdout/stderr во время загрузки моделей
- Замер времени выполнения каждого этапа
- Оптимизация памяти при работе с временными файлами

## Зависимости

- **whisperx** - основная библиотека транскрипции
- **pyannote.audio** - сегментация и диаризация
- **librosa** - обработка аудио
- **torch** - машинное обучение
- **pydantic** - валидация конфигурации

## Конфигурационные параметры

### Whisper Config
- `whisper_arch` - архитектура модели (large-v2, medium, etc.)
- `compute_type` - тип вычислений (int8_float16, float16)
- `device` - устройство (cuda, cpu)
- `language` - язык для транскрипции

### Align Config
- `language_code` - код языка для выравнивания
- `model_name` - модель для выравнивания (опционально)
- `device` - устройство для выравнивания

### Segmentation Config
- `model` - модель для сегментации (pyannote/segmentation)
- `batch_size` - размер батча
- `step` - шаг сегментации
- `peak_config` - настройки поиска пиков 