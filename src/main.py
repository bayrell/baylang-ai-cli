"""Точка входа BayLang AI: интерактивный чат и одноразовый запрос.

Запуск из корня репозитория::

    python -m src.main --prompt "Скажи привет"
    python -m src                # интерактивный режим (через src/__main__.py)

Конфигурация:

* ключ API читается из переменной окружения ``OPENROUTER_API_KEY``;
* модель — из ``OPENROUTER_MODEL`` или аргумента ``--model``
  (приоритет: аргумент > переменная окружения > значение по умолчанию).

Коды возврата: ``0`` — успех, ``1`` — ошибка выполнения,
``2`` — ошибка конфигурации.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import time
from datetime import datetime
from typing import Optional, Sequence

from .ai import (
    Agent,
    BayLangError,
    ConfigurationError,
    OpenRouterProvider,
    ProviderError,
    TextMessage,
    Tool,
    ToolRegistry,
)

__all__ = [
    "build_agent",
    "run_interactive",
    "run_once",
    "parse_args",
    "main",
    "mask_api_key",
]

APP_NAME = "BayLang AI"
APP_VERSION = "0.1.0"
DEFAULT_MODEL = "openai/gpt-4o-mini"
ENV_API_KEY = "OPENROUTER_API_KEY"
ENV_MODEL = "OPENROUTER_MODEL"

EXIT_OK = 0
EXIT_RUNTIME_ERROR = 1
EXIT_CONFIG_ERROR = 2

SYSTEM_PROMPT = (
    "Ты — полезный ассистент BayLang AI. Отвечай кратко и по делу, "
    "по-русски, если пользователь не просит иначе."
)

HELP_TEXT = """\
Команды:
  help          — показать эту справку
  clear         — очистить контекст диалога
  model <имя>   — сменить модель на лету (например: model openai/gpt-4o-mini)
  exit, quit    — завершить работу (также Ctrl+C и Ctrl+D)
Любая другая строка отправляется модели как запрос."""

logger = logging.getLogger(__name__)

_LOOP: Optional[asyncio.AbstractEventLoop] = None


# ---------------------------------------------------------------------------
# Аргументы командной строки
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Разобрать аргументы командной строки.

    Args:
        argv: Аргументы без имени программы; ``None`` — ``sys.argv[1:]``.

    Returns:
        Пространство имён с аргументами.
    """
    parser = argparse.ArgumentParser(
        prog="baylang",
        description=f"{APP_NAME} {APP_VERSION} — чат с LLM через OpenRouter",
    )
    parser.add_argument(
        "--prompt",
        help="одноразовый запрос: выполнить и завершиться (без интерактивного режима)",
    )
    parser.add_argument(
        "--model",
        help=f"модель OpenRouter (по умолчанию: {DEFAULT_MODEL}, env {ENV_MODEL})",
    )
    parser.add_argument(
        "--max-iters",
        type=int,
        default=5,
        help="максимум итераций tool-calling (по умолчанию: 5)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="температура генерации (по умолчанию: 0.7)",
    )
    parser.add_argument(
        "--no-tools",
        action="store_true",
        help="не регистрировать демонстрационные инструменты",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="подробное логирование и статистика ответов",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"{APP_NAME} {APP_VERSION}",
    )
    return parser.parse_args(argv)


def mask_api_key(key: str) -> str:
    """Замаскировать API-ключ для безопасного вывода в логи.

    Args:
        key: Исходный ключ.

    Returns:
        Ключ вида ``sk-o****************1234``; короткие ключи целиком
        заменяются звёздочками.
    """
    if not key:
        return ""
    if len(key) <= 8:
        return "*" * len(key)
    return f"{key[:4]}{'*' * (len(key) - 8)}{key[-4:]}"


# ---------------------------------------------------------------------------
# Сборка агента
# ---------------------------------------------------------------------------


def _build_tools() -> list[Tool]:
    """Создать демонстрационные инструменты агента.

    Returns:
        Список инструментов ``get_current_time`` и ``web_search`` (заглушка).
    """

    def get_current_time() -> str:
        """Текущие дата и время на сервере в формате ISO-8601."""
        return datetime.now().isoformat(timespec="seconds")

    def web_search_stub(query: str = "") -> str:
        """Демонстрационная заглушка веб-поиска."""
        return f"[заглушка поиска] Запрос {query!r} обработан локально; реальный поиск не выполнялся."

    return [
        Tool(
            name="get_current_time",
            description="Возвращает текущие дату и время на сервере.",
            parameters_schema={"type": "object", "properties": {}, "required": []},
            handler=get_current_time,
        ),
        Tool(
            name="web_search",
            description="Демонстрационная заглушка веб-поиска: подтверждает запрос, не выполняя его.",
            parameters_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Поисковый запрос"}
                },
                "required": ["query"],
            },
            handler=web_search_stub,
        ),
    ]


def build_agent(args: argparse.Namespace) -> Agent:
    """Собрать агента: провайдер, инструменты, параметры.

    Ключ API читается из переменной окружения ``OPENROUTER_API_KEY``,
    модель — из ``--model`` или ``OPENROUTER_MODEL``.

    Args:
        args: Аргументы командной строки (:func:`parse_args`).

    Returns:
        Готовый объект :class:`~src.ai.Agent`.

    Raises:
        ConfigurationError: Если ключ API не задан.
    """
    api_key = os.environ.get(ENV_API_KEY, "").strip()
    if not api_key:
        raise ConfigurationError(
            f"Переменная окружения {ENV_API_KEY} не задана. "
            f"Получите ключ на https://openrouter.ai/keys и выполните: "
            f"export {ENV_API_KEY}=sk-or-..."
        )
    model = (args.model or os.environ.get(ENV_MODEL, "").strip() or DEFAULT_MODEL).strip()
    provider = OpenRouterProvider(
        api_key=api_key,
        model_name=model,
        temperature=args.temperature,
    )
    registry = ToolRegistry()
    if not args.no_tools:
        for tool in _build_tools():
            registry.register(tool)
    return Agent(
        provider=provider,
        tools=registry,
        max_iters=args.max_iters,
        system_prompt=SYSTEM_PROMPT,
    )


# ---------------------------------------------------------------------------
# Режимы работы
# ---------------------------------------------------------------------------


def _get_loop() -> asyncio.AbstractEventLoop:
    """Вернуть (или создать) долгоживущий event loop приложения."""
    global _LOOP
    if _LOOP is None or _LOOP.is_closed():
        _LOOP = asyncio.new_event_loop()
        asyncio.set_event_loop(_LOOP)
    return _LOOP


def _usage_suffix(agent: Agent) -> str:
    """Собрать суффикс статистики (токены) для verbose-вывода.

    Returns:
        Строка вида ``"; prompt tokens: 10; completion tokens: 5"``.
    """
    response = agent.last_response
    if not response or not response.usage:
        return ""
    parts = []
    for key, label in (("prompt_tokens", "prompt"), ("completion_tokens", "completion")):
        value = response.usage.get(key)
        if isinstance(value, int):
            parts.append(f"{label} tokens: {value}")
    return ("; " + "; ".join(parts)) if parts else ""


def _should_retry(exc: ProviderError) -> bool:
    """Определить, стоит ли повторять запрос после ошибки.

    Повтор допускается при сетевых сбоях (``status_code is None``)
    и серверных ошибках HTTP (5xx).
    """
    return exc.status_code is None or exc.status_code >= 500


def _send_with_retry(agent: Agent, prompt_text: str, attempts: int = 2) -> str:
    """Отправить реплику пользователя с одной повторной попыткой при сбое.

    Args:
        agent: Агент.
        prompt_text: Текст реплики пользователя.
        attempts: Максимальное число попыток (по умолчанию 2).

    Returns:
        Текст финального ответа модели.

    Raises:
        ProviderError: Последняя попытка завершилась ошибкой.
    """
    agent.context.add_message(TextMessage.user(prompt_text))
    last_exc: Optional[ProviderError] = None
    loop = _get_loop()
    for attempt in range(1, attempts + 1):
        try:
            return loop.run_until_complete(agent.send(agent.context))
        except ProviderError as exc:
            last_exc = exc
            if attempt >= attempts or not _should_retry(exc):
                break
            logger.warning(
                "Попытка %d завершилась ошибкой (%s); повторяю запрос", attempt, exc
            )
    assert last_exc is not None
    raise last_exc


def run_once(agent: Agent, prompt: str, args: argparse.Namespace) -> str:
    """Выполнить одноразовый запрос (режим ``--prompt``).

    Args:
        agent: Агент.
        prompt: Текст запроса пользователя.
        args: Аргументы командной строки.

    Returns:
        Текст ответа модели; печать выполняется вызывающим кодом.
    """
    started = time.perf_counter()
    text = _send_with_retry(agent, prompt)
    elapsed = time.perf_counter() - started
    if args.verbose:
        print(
            f"[verbose] время ответа: {elapsed:.2f} с{_usage_suffix(agent)}",
            file=sys.stderr,
        )
    return text


def run_interactive(agent: Agent, args: argparse.Namespace) -> int:
    """Интерактивный режим: цикл чтения строк со стандартного ввода.

    Команды: ``help``, ``clear``, ``model <имя>``, ``exit``/``quit``.
    Ошибки API не роняют приложение: сообщение выводится, диалог продолжается.

    Args:
        agent: Агент.
        args: Аргументы командной строки.

    Returns:
        Код выхода (0 — корректное завершение).
    """
    print(f"{APP_NAME} {APP_VERSION}. Справка: help; очистка контекста: clear; выход: exit.")
    loop = _get_loop()
    while True:
        try:
            line = input("you> ")
        except (EOFError, KeyboardInterrupt):
            print("\nДо встречи! 👋")
            return EXIT_OK

        line = line.strip()
        if not line:
            continue

        parts = line.split(maxsplit=1)
        command = parts[0].lower()
        rest = parts[1].strip() if len(parts) > 1 else ""

        if command in ("exit", "quit"):
            print("До встречи! 👋")
            return EXIT_OK
        if command == "help":
            print(HELP_TEXT)
            continue
        if command == "clear":
            agent.reset()
            print("Контекст диалога очищен.")
            continue
        if command == "model":
            if not rest:
                print("Использование: model <имя модели>")
                continue
            agent.provider.model_name = rest
            print(f"Модель изменена: {rest}")
            continue

        started = time.perf_counter()
        agent.context.add_message(TextMessage.user(line))
        try:
            text = loop.run_until_complete(agent.send(agent.context))
        except ProviderError as exc:
            print(f"assistant> Ошибка запроса: {exc}")
            print("Проверьте сеть и API-ключ, затем попробуйте ещё раз.")
            continue
        except BayLangError as exc:
            print(f"assistant> Ошибка: {exc}")
            continue
        elapsed = time.perf_counter() - started
        print(f"assistant> {text}")
        if args.verbose:
            print(f"[verbose] время ответа: {elapsed:.2f} с{_usage_suffix(agent)}")


# ---------------------------------------------------------------------------
# Оркестратор
# ---------------------------------------------------------------------------


def _shutdown(agent: Agent, loop: asyncio.AbstractEventLoop) -> None:
    """Закрыть HTTP-клиент провайдера и event loop (не должен падать)."""
    provider = agent.provider
    aclose = getattr(provider, "aclose", None)
    if aclose is not None:
        try:
            loop.run_until_complete(aclose())
        except Exception:  # noqa: BLE001 — очистка не должна ронять программу
            logger.debug("Не удалось закрыть HTTP-клиент провайдера", exc_info=True)
    if not loop.is_closed():
        loop.close()


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Оркестратор приложения: парсинг аргументов, сборка агента, режим работы.

    Args:
        argv: Аргументы командной строки; ``None`` — ``sys.argv[1:]``.

    Returns:
        Код выхода процесса: ``0`` — успех, ``1`` — ошибка выполнения,
        ``2`` — ошибка конфигурации.
    """
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        agent = build_agent(args)
    except ConfigurationError as exc:
        print(f"Ошибка конфигурации: {exc}", file=sys.stderr)
        return EXIT_CONFIG_ERROR
    except BayLangError as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return EXIT_CONFIG_ERROR

    if args.verbose:
        api_key = os.environ.get(ENV_API_KEY, "")
        logger.debug(
            "Модель: %s; ключ API: %s",
            agent.provider.get_model_name(),
            mask_api_key(api_key),
        )

    loop = _get_loop()
    try:
        if args.prompt:
            text = run_once(agent, args.prompt, args)
            print(text)
            return EXIT_OK
        return run_interactive(agent, args)
    except KeyboardInterrupt:
        print("\nДо встречи! 👋")
        return EXIT_OK
    except ProviderError as exc:
        print(f"Ошибка выполнения: {exc}", file=sys.stderr)
        return EXIT_RUNTIME_ERROR
    except BayLangError as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return EXIT_RUNTIME_ERROR
    finally:
        _shutdown(agent, loop)


if __name__ == "__main__":
    sys.exit(main())
