# syntax=docker/dockerfile:1
# BayLang AI — контейнер с CLI-приложением и библиотекой (OpenRouter chat-completions).
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Зависимости кешируются отдельным слоем: пересобирается только при изменении requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Исходники приложения
COPY src/ ./src/

# Конфигурация передаётся через переменные окружения:
#   OPENROUTER_API_KEY — ключ OpenRouter (обязательна)
#   OPENROUTER_MODEL   — модель (необязательна, по умолчанию openai/gpt-4o-mini)
#
# Промпт и история диалогов хранятся в /root/.baylang (монтируйте том для сохранности):
#   docker run --rm -e OPENROUTER_API_KEY -v "$HOME/.baylang:/root/.baylang" baylang-ai

ENTRYPOINT ["python", "-m", "src.main"]
CMD ["--help"]
