"""Точка входа BayLang AI: интерактивный чат и одноразовый запрос.

Запуск из корня репозитория::

    python -m src.main --prompt "Скажи привет"
    python -m src                # интерактивный режим (через src/__main__.py)

Конфигурация (хранится в переменных окружения):

* ключ API — ``OPENROUTER_API_KEY`` (обязательна);
* модель — ``OPENROUTER_MODEL`` или аргумент ``--model``
  (приоритет: аргумент > переменная окружения > значение по умолчанию).

Домашняя папка приложения (``~/.baylang``):

* ``prompt.txt`` — системный промпт (создаётся с промптом по умолчанию,
  если файла нет; путь можно переопределить через ``--prompt-file``);
* ``history/`` — сохранённые истории диалогов (JSON, имя по умолчанию —
  метка времени, например ``20250101_120000.json``).

При запуске история всегда чистая; загрузить сохранённую историю можно
флагом ``--history <имя>`` или командой ``load <имя>`` в интерактивном
режиме. Список последних историй выводится командой ``histories`` или
флагом ``--list-histories``.

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
from pathlib import Path
from typing import Optional, Sequence

from .ai import (
    Agent,
    BayLangError,
    ConfigurationError,
    Context,
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
    "ensure_app_dirs",
    "load_system_prompt",
    "list_histories",
    "save_history",
    "load_history",
    "print_histories",
]

APP_NAME = "BayLang AI"
APP_VERSION = "0.2.0"
DEFAULT_MODEL = "openai/gpt-4o-mini"
ENV_API_KEY = "OPENROUTER_API_KEY"
ENV_MODEL = "OPENROUTER_MODEL"

EXIT_OK = 0
EXIT_RUNTIME_ERROR = 1
EXIT_CONFIG_ERROR = 2

# Домашняя папка приложения: промпт и истории диалогов.
APP_DIR = Path.home() / ".baylang"
DEFAULT_PROMPT_FILE = APP_DIR / "prompt.txt"
HISTORY_DIR = APP_DIR / "history"
HISTORY_LIST_LIMIT = 10

SYSTEM_PROMPT = (
    "Ты — полезный ассистент BayLang AI. Отвечай кратко и по делу, "
    "по-русски, если пользователь не просит иначе."
)

HELP_TEXT = """\
Команды:
  help                — показать эту справку
  clear               — очистить контекст диалога
  model <имя>         — сменить модель на лету (например: model openai/gpt-4o-mini)
  save [имя]          — сохранить историю (имя по умолчанию — метка времени)
  load <имя>          — загрузить историю из ~/.baylang/history
  histories           — список последних историй
  exit, quit          — завершить работу (также Ctrl+C и Ctrl+D)
Любая другая строка отправляется модели как запрос.
Файл промпта: ~/.baylang/prompt.txt; история: ~/.baylang/history/."""

logger = logging.getLogger(__name__)

_LOOP: Optional[asyncio.AbstractEventLoop] = None


# ---------------------------------------------------------------------------
# Домашняя папка, промпт и истории
# ---------------------------------------------------------------------------


def ensure_app_dirs() -> None:
    """Создать домашнюю папку приложения и каталог историй.

    Если файл ``prompt.txt`` отсутствует, записывается промпт по умолчанию,
    чтобы пользователь мог отредактировать его перед запуском.
    """
    APP_DIR.mkdir(parents=True, exist_ok=True)
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    if not DEFAULT_PROMPT_FILE.exists():
        try:
            DEFAULT_PROMPT_FILE.write_text(SYSTEM_PROMPT + "\n", encoding="utf-8")
        except OSError as exc:
            logger.warning("Не удалось создать %s: %s", DEFAULT_PROMPT_FILE, exc)


def load_system_prompt(path: Optional[Path] = None) -> str:
    """Загрузить системный промпт из файла.

    По умолчанию читается ``~/.baylang/prompt.txt``. Если файл отсутствует
    или пуст, возвращается промпт по умолчанию (:data:`SYSTEM_PROMPT`).

    Args:
        path: Путь к файлу промпта; ``None`` — путь по умолчанию.

    Returns:
        Текст системного промпта.
    """
    file_path = Path(path).expanduser() if path else DEFAULT_PROMPT_FILE
    try:
        text = file_path.read_text(encoding="utf-8").strip()
    except OSError:
        return SYSTEM_PROMPT
    return text or SYSTEM_PROMPT


def _normalize_history_name(name: str) -> str:
    """Нормализовать имя истории в безопасное имя файла ``*.json``.

    Args:
        name: Имя истории (без расширения или с ``.json``).

    Returns:
        Имя файла вида ``<name>.json``.

    Raises:
        ValueError: Если имя пустое или содержит недопустимые символы.
    """
    name = (name or "").strip()
    if not name:
        raise ValueError("Имя истории не может быть пустым")
    if any(sep in name for sep in ("/", "\\")) or name in (".", "..") or name.startswith("."):
        raise ValueError(f"Недопустимое имя истории: {name!r}")
    if not name.lower().endswith(".json"):
        name += ".json"
    return name


def list_histories(limit: int = HISTORY_LIST_LIMIT) -> list[Path]:
    """Вернуть пути к последним файлам истории (новые — первыми).

    Args:
        limit: Максимальное число файлов.

    Returns:
        Список путей к JSON-файлам истории; пустой список, если история
        ещё не создавалась.
    """
    if not HISTORY_DIR.is_dir():
        return []
    files = [p for p in HISTORY_DIR.glob("*.json") if p.is_file()]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files[: max(int(limit), 0)]


def save_history(agent: Agent, name: Optional[str] = None) -> Path:
    """Сохранить текущий контекст агента в файл истории.

    Args:
        agent: Агент, чей контекст сохраняется.
        name: Имя истории; ``None`` — метка времени в формате
            ``YYYYMMDD_HHMMSS`` (timestamp).

    Returns:
        Путь к сохранённому файлу истории.

    Raises:
        ValueError: Если имя истории некорректно.
        OSError: Если файл не удалось записать.
    """
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    if name:
        filename = _normalize_history_name(name)
    else:
        filename = datetime.now().strftime("%Y%m%d_%H%M%S") + ".json"
    path = HISTORY_DIR / filename
    path.write_text(agent.context.to_json(indent=2), encoding="utf-8")
    return path


def load_history(agent: Agent, name: str) -> int:
    """Загрузить историю по имени и подставить её контексту агента.

    Args:
        agent: Агент, контекст которого заменяется содержимым истории.
        name: Имя истории (в ``~/.baylang/history``).

    Returns:
        Количество загруженных сообщений.

    Raises:
        FileNotFoundError: Если файл истории не найден.
        ValueError: Если имя некорректно или файл повреждён.
    """
    filename = _normalize_history_name(name)
    path = HISTORY_DIR / filename
    if not path.is_file():
        raise FileNotFoundError(f"История {filename!r} не найдена в {HISTORY_DIR}")
    try:
        agent.context = Context.from_json(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise ValueError(f"Файл истории {filename!r} повреждён: {exc}") from exc
    return len(agent.context)


def print_histories(limit: int = HISTORY_LIST_LIMIT) -> None:
    """Вывести список последних историй (имя, дата, размер)."""
    files = list_histories(limit)
    if not files:
        print(f"Историй пока нет: {HISTORY_DIR}")
        return
    print(f"Последние истории ({HISTORY_DIR}):")
    for path in files:
        stat = path.stat()
        mtime = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        print(f"  {path.stem:<24} {mtime}  {stat.st_size} байт")


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
        "--prompt-file",
        help=f"файл системного промпта (по умолчанию: {DEFAULT_PROMPT_FILE})",
    )
    parser.add_argument(
        "--history",
        metavar="ИМЯ",
        help=(
            "загрузить сохранённую историю из ~/.baylang/history/<имя>.json "
            "перед стартом (по умолчанию — чистая история)"
        ),
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="сохранить историю при завершении (имя — метка времени)",
    )
    parser.add_argument(
        "--list-histories",
        action="store_true",
        help="вывести список последних историй и завершиться",
    )
    parser.add_argument(
        "--max-iters",
        type=int,
        default=5,
        help="максимум итераций tool (по умолчанию: 5)",
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


def build_agent(args: argparse.Namespace) -> Agent:
    """Собрать агента: провайдер, инструменты, параметры.

    Ключ API читается из переменной окружения ``OPENROUTER_API_KEY``,
    модель — из ``--model`` или ``OPENROUTER_MODEL``. Системный промпт
    загружается из файла (по умолчанию ``~/.baylang/prompt.txt``).

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
    prompt_file = getattr(args, "prompt_file", None)
    system_prompt = load_system_prompt(Path(prompt_file).expanduser() if prompt_file else None)
    provider = OpenRouterProvider(
        api_key=api_key,
        model_name=model,
        temperature=args.temperature,
    )
    registry = ToolRegistry()
    return Agent(
        provider=provider,
        tools=registry,
        max_iters=args.max_iters,
        system_prompt=system_prompt,
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

    Команды: ``help``, ``clear``, ``model <имя>``, ``save [имя]``,
    ``load <имя>``, ``histories``, ``exit``/``quit``. Ошибки API не роняют
    приложение: сообщение выводится, диалог продолжается.

    Args:
        agent: Агент.
        args: Аргументы командной строки.

    Returns:
        Код выхода (0 — корректное завершение).
    """
    print(
        f"{APP_NAME} {APP_VERSION}. Справка: help; история: histories/save/load; "
        "очистка контекста: clear; выход: exit."
    )
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
        if command == "save":
            try:
                path = save_history(agent, rest or None)
            except ValueError as exc:
                print(f"Ошибка: {exc}")
                continue
            except OSError as exc:
                print(f"Не удалось сохранить историю: {exc}")
                continue
            print(f"История сохранена: {path}")
            continue
        if command == "load":
            if not rest:
                print("Использование: load <имя> (список историй: histories)")
                continue
            try:
                count = load_history(agent, rest)
            except (FileNotFoundError, ValueError) as exc:
                print(f"Не удалось загрузить историю: {exc}")
                continue
            print(f"История загружена: {count} сообщений.")
            continue
        if command == "histories":
            print_histories()
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

    История по умолчанию чистая; загрузка выполняется по ``--history``,
    сохранение по ``--save`` или команде ``save`` в интерактивном режиме.

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

    ensure_app_dirs()

    if args.list_histories:
        print_histories()
        return EXIT_OK

    try:
        agent = build_agent(args)
    except ConfigurationError as exc:
        print(f"Ошибка конфигурации: {exc}", file=sys.stderr)
        return EXIT_CONFIG_ERROR
    except BayLangError as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return EXIT_CONFIG_ERROR

    if args.history:
        try:
            count = load_history(agent, args.history)
        except (FileNotFoundError, ValueError) as exc:
            print(f"Ошибка загрузки истории: {exc}", file=sys.stderr)
            return EXIT_CONFIG_ERROR
        print(
            f"Загружена история '{args.history}': {count} сообщений.",
            file=sys.stderr,
        )

    if args.verbose:
        api_key = os.environ.get(ENV_API_KEY, "")
        logger.debug(
            "Модель: %s; ключ API: %s; промпт: %s",
            agent.provider.get_model_name(),
            mask_api_key(api_key),
            DEFAULT_PROMPT_FILE,
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
        if args.save:
            try:
                path = save_history(agent, name=None)
                print(f"История сохранена: {path}", file=sys.stderr)
            except (OSError, ValueError) as exc:
                print(f"Не удалось сохранить историю: {exc}", file=sys.stderr)
        _shutdown(agent, loop)


if __name__ == "__main__":
    sys.exit(main())
