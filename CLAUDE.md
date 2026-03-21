# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Обзор проекта

MeetScribe — CLI-приложение для транскрибации встреч с диаризацией спикеров. Комбинирует speech-to-text модели, детекцию голосовой активности, нейросетевые эмбеддинги спикеров и кластеризацию для разделения и идентификации участников.

## Команды разработки

```bash
# Установка
uv venv
uv pip install -e ".[dev]"

# Тесты
uv run pytest

# Линтинг и форматирование
uv run ruff check src/
uv run ruff format src/

# Запуск CLI
uv run meetscribe --help
```

## Архитектура

### Точка входа

`src/meetscribe/cli.py` — CLI на argparse с подкомандами:
- `transcribe` — основная транскрибация с диаризацией
- `enroll` — регистрация известных спикеров
- `extract` — извлечение аудиодорожек из видео
- `extract-samples` — извлечение семплов для последующей регистрации
- `list-speakers` — список зарегистрированных спикеров
- `info` — информация о конфигурации

### Pipeline (`src/meetscribe/pipeline/`)

Модульный конвейер обработки аудио:

1. **VAD (vad.py)** — Voice Activity Detection
   - Детекция участков с речью, фильтрация тишины и шумов
   - Параметры: пороги начала/окончания речи, минимальная длительность сегмента
   - Уменьшает галлюцинации транскрипции за счёт передачи только речевых участков

2. **Embeddings (embeddings.py)** — Извлечение спикерных эмбеддингов
   - Нейросетевая модель (ECAPA-TDNN архитектура) преобразует аудио в вектор фиксированной размерности
   - Эмбеддинги используются для сравнения голосов через косинусное сходство
   - Кэширование моделей через HuggingFace Hub

3. **Diarization (diarization.py)** — Кластеризация спикеров
   - Группировка сегментов по спикерам на основе эмбеддингов
   - Спектральная кластеризация с автоопределением числа спикеров
   - Возвращает `DiarizedSegment` с cluster_id

4. **Identification (identification.py)** — Идентификация известных спикеров
   - Сравнение центроидов кластеров с сохранёнными voiceprints
   - Хранение voiceprints в JSON (путь из config)
   - Порог косинусного сходства для match/unknown

5. **Transcription (transcription.py)** — Speech-to-text
   - Whisper-модели разных размеров (tiny → large)
   - Обработка по VAD-сегментам для точности
   - Назначение спикера по временному пересечению с диаризованными сегментами

### Конфигурация (`src/meetscribe/config.py`)

Платформозависимые директории:
- Windows: `%LOCALAPPDATA%/meetscribe`
- macOS: `~/Library/Application Support/meetscribe`
- Linux: `~/.local/share/meetscribe` (XDG_DATA_HOME)

Поддиректории: models (кэш), voiceprints, samples, logs.

## Ключевые концепции

### Обработка аудио

- Все модели работают с 16kHz mono
- Аудио загружается через torchaudio, ресемплится при необходимости
- FFmpeg используется для извлечения дорожек из видео (bundled через Python-пакет)
- Временные файлы создаются в tempfile для промежуточных результатов

### Пайплайн обработки

```
Video/Audio → Extract tracks → VAD → Embeddings → Clustering → Identification → Transcription → Markdown
```

Многодорожечная обработка:
- Видео с несколькими аудиотреками (трек 1 = ведущий, трек 2 = гости)
- Именованные треки (`--track1 "Name"`) vs автоматическая диаризация
- Неименованные треки проходят через полный пайплайн

### Модели и кэширование

- Ленивая загрузка моделей (загружаются только при использовании)
- Кэширование весов через HuggingFace Hub
- Device detection (CUDA/CPU) для тензорных операций
- Windows: настройка путей к DLL для NVIDIA CUDA

### Dataclasses

- `SpeechSegment` — временной интервал с речью (start_ms, end_ms)
- `DiarizedSegment` — сегмент с cluster_id и эмбеддингом
- `TranscriptSegment` — финальный результат с текстом и спикером
- `SpeakerMatch` — результат идентификации (имя + confidence)

## Стиль кода

- Python 3.12+ с union syntax (`X | None`)
- Type hints везде
- Ruff: line-length 100, rules E/F/I/UP
- Модульная архитектура с чёткими интерфейсами между компонентами

## Windows-специфика

- Настройка NVIDIA DLL paths для CUDA
- Console color support через colorama
- Bundled FFmpeg не требует системной установки
- HuggingFace cache: symlinks отключены, используется copy strategy

## CLI UX

- ANSI-цвета для форматированного вывода
- Прогресс-индикаторы и эмодзи для этапов
- Иерархия step/substep для визуальной структуры
- Логирование: файл (DEBUG) + консоль (ERROR)
- Форматирование времени в человекочитаемом виде
