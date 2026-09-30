#!/usr/bin/env bash
#
# build.sh — сборка и запуск Docker-образа BayLang AI.
#
# Использование:
#   ./build.sh build [tag]           # docker build -t baylang-ai:<tag>
#   ./build.sh run [tag] [args...]   # запуск контейнера (аргументы приложению)
#   ./build.sh test [tag]            # pytest внутри контейнера
#   ./build.sh clean [tag]           # удалить образ
#   ./build.sh help                  # справка
#
# Переменные окружения:
#   IMAGE_NAME           имя образа (по умолчанию: baylang-ai)
#   IMAGE_TAG            тег образа (по умолчанию: latest)
#   OPENROUTER_API_KEY   ключ API OpenRouter (нужен для команды run)
#   OPENROUTER_MODEL     модель (необязательно)
#
set -euo pipefail

IMAGE_NAME="bayrell/baylang-ai-command"
IMAGE_TAG="1.0.0"

usage() {
    cat <<EOF
Использование: ./build.sh <команда> [аргументы]

Команды:
  build [tag]     — собрать образ (по умолчанию тег: ${IMAGE_TAG})
  run [tag] [..]  — запустить контейнер; аргументы передаются приложению
  test [tag]      — выполнить pytest внутри контейнера
  clean [tag]     — удалить образ
  help            — показать эту справку

Переменные окружения:
  IMAGE_NAME           имя образа (по умолчанию: ${IMAGE_NAME})
  IMAGE_TAG            тег образа (по умолчанию: ${IMAGE_TAG})
  OPENROUTER_API_KEY   ключ API OpenRouter (нужен для команды run)
  OPENROUTER_MODEL     модель (необязательно)

Команда run монтирует ~/.baylang в контейнер,
чтобы сохранять prompt.txt и историю диалогов между запусками.

Примеры:
  ./build.sh build
  ./build.sh build 1.0
  ./build.sh run latest --prompt "Скажи привет"
  ./build.sh run --list-histories
EOF
}

# Взять первый аргумент как тег образа, если он не начинается с дефиса
take_tag() {
    if [ $# -gt 0 ] && [[ "${1}" != -* ]]; then
        TAG_VALUE="${1}"
        shift
    else
        TAG_VALUE="${IMAGE_TAG}"
    fi
    REMAINING_ARGS=("$@")
}

cmd="${1:-help}"
shift || true

case "${cmd}" in
    build)
        take_tag "$@"
        echo "==> Собираю образ ${IMAGE_NAME}:${TAG_VALUE}"
        docker build -t "${IMAGE_NAME}:${TAG_VALUE}" .
        ;;
    run)
        take_tag "$@"
        echo "==> Запускаю ${IMAGE_NAME}:${TAG_VALUE}"
        docker run --rm \
            -e OPENROUTER_API_KEY \
            -e OPENROUTER_MODEL \
            -v "${HOME}/.baylang:/root/.baylang" \
            "${IMAGE_NAME}:${TAG_VALUE}" "${REMAINING_ARGS[@]}"
        ;;
    test)
        take_tag "$@"
        echo "==> Тесты в образе ${IMAGE_NAME}:${TAG_VALUE}"
        set +e
        docker run --rm "${IMAGE_NAME}:${TAG_VALUE}" pytest -q
        code=$?
        set -e
        if [ "${code}" -eq 5 ]; then
            echo "Тесты не найдены — считаю пройденными (pytest exit code 5)"
            exit 0
        fi
        exit "${code}"
        ;;
    clean)
        take_tag "$@"
        echo "==> Удаляю образ ${IMAGE_NAME}:${TAG_VALUE}"
        docker rmi "${IMAGE_NAME}:${TAG_VALUE}" || true
        ;;
    help|-h|--help)
        usage
        ;;
    *)
        echo "Неизвестная команда: ${cmd}" >&2
        echo >&2
        usage >&2
        exit 2
        ;;
esac
