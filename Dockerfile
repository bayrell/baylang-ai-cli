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

CMD ["bash"]
