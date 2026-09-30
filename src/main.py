#!/usr/bin/env python3

"""Точка входа BayLang AI: интерактивный чат и одноразовый запрос.

Запуск из корня репозитория::

    python -m src.main --prompt "Скажи привет"
    python -m src                # интерактивный режим (через src/__main__.py)
    python src/main.py           # прямой запуск файла (fallback-импорт)

Конфигурация (хранится в переменных окружения):

* ключ API — ``OPENROUTER_API_KEY`` (обязательна);
* модель — ``OPENROUTER_MODEL`` или аргумент ``--model``
  (приоритет: аргумент > переменная окружения > значение по умолчанию).

Домашняя папка приложения (``~/.baylang``):

* ``prompt.txt`` — системный промпт (создаётся с промптом по умолчанию,
  если файла нет; путь можно переопределить через ``--prompt-file``);
* ``history/`` — сохранённые истории диалогов (JSON pretty, имя файла —
  метка времени сессии, например ``20250101_120000.json``);
* ``baylang.log`` — журнал работы приложения (с ротацией: до 3 бэкапов
  по 1 МБ, в файл пишутся сообщения начиная с INFO, независимо от
  ``--verbose``; в консоль — WARNING, либо DEBUG при ``--verbose``).

По умолчанию всегда запускается **новая сессия**; сохранённую историю
можно подхватить флагом ``--history <имя>`` (алиас ``--load <имя>``)
или командой ``load <имя>`` в интерактивном режиме. История сохраняется
автоматически и всегда: после каждого ответа модели и ещё раз при
завершении работы. Список последних историй выводится командой
``histories`` или флагом ``--list-histories``.

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
from logging.handlers import RotatingFileHandler
from dotenv import load_dotenv
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

from ai import (  # type: ignore[no-redef]
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
    "parse_args",
    "main",
    "mask_api_key",
    "ensure_app_dirs",
    "setup_logging",
    "load_system_prompt",
    "list_histories",
    "save_history",
    "load_history",
    "print_histories",
]

load_dotenv()

APP_NAME = "BayLang AI"
APP_VERSION = "1.0.0"
DEFAULT_MODEL = "openrouter/auto"
ENV_API_KEY = "OPENROUTER_API_KEY"
ENV_MODEL = "OPENROUTER_MODEL"

EXIT_OK = 0
EXIT_RUNTIME_ERROR = 1
EXIT_CONFIG_ERROR = 2

# Домашняя папка приложения: промпт, истории диалогов и журнал.
APP_DIR = Path.home() / ".baylang"
DEFAULT_PROMPT_FILE = APP_DIR / "prompt.txt"
HISTORY_DIR = APP_DIR / "history"
HISTORY_LIST_LIMIT = 10
LOG_FILE = APP_DIR / "baylang.log"
LOG_MAX_BYTES = 1_000_000
LOG_BACKUP_COUNT = 3
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"

SYSTEM_PROMPT = (
    "Ты — полезный ассистент BayLang AI. Отвечай кратко и по делу, "
    "по-русски, если пользователь не просит иначе."
)

HELP_TEXT = """\
Команды:
  help                — показать эту справку
  clear               — очистить контекст диалога
  model <имя>         — сменить модель на лету (например: model openai/gpt-4o-mini)
  save [имя]          — сохранить историю под другим именем (по умолчанию — метка времени)
  load <имя>          — загрузить историю из ~/.baylang/history (флаг: --load/--history)
  histories           — список последних историй
  exit, quit          — завершить работу (также Ctrl+C и Ctrl+D)
Любая другая строка отправляется модели как запрос.
История сессии сохраняется автоматически после каждого ответа и при выходе.
Файл промпта: ~/.baylang/prompt.txt; история: ~/.baylang/history/;
журнал: ~/.baylang/baylang.log."""

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Домашняя папка, промпт, журнал и истории
# ---------------------------------------------------------------------------


def ensure_app_dirs() -> None:
    """Создать домашнюю папку приложения и каталог историй.

    Если файл ``prompt.txt`` отсутствует, записывается промпт по умолчанию,
    чтобы пользователь мог отредактировать его перед запуском. Каталог
    создаётся и для журнала ``baylang.log`` (лежит в самой домашней папке).
    """
    APP_DIR.mkdir(parents=True, exist_ok=True)
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    if not DEFAULT_PROMPT_FILE.exists():
        try:
            DEFAULT_PROMPT_FILE.write_text(SYSTEM_PROMPT + "\n", encoding="utf-8")
        except OSError as exc:
            logger.warning("Не удалось создать %s: %s", DEFAULT_PROMPT_FILE, exc)


def setup_logging(verbose: bool = False) -> Path:
    """Настроить логирование: консоль + файл журнала в домашней папке.

    Args:
        verbose: ``True`` — дублировать DEBUG-сообщения в консоль и в файл.

    Returns:
        Путь к открытому файлу журнала (или к ``LOG_FILE``, если файл
        открыть не удалось — ошибка не прерывает работу приложения).
    """
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()

    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG if verbose else logging.WARNING)
    console.setFormatter(logging.Formatter(LOG_FORMAT))
    root.addHandler(console)

    try:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            LOG_FILE,
            maxBytes=LOG_MAX_BYTES,
            backupCount=LOG_BACKUP_COUNT,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
        root.addHandler(file_handler)
    except OSError as exc:
        root.warning("Не удалось открыть журнал %s: %s", LOG_FILE, exc)
    return LOG_FILE


def load_system_prompt(path: Optional[Path] = None) -> str:
    """Загрузить системный промпт из файла.

    По умолчанию читается ``~/.baylang/prompt.txt``. Если файл отсутствует
    или пуст, возвращается промпт по умолчанию

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


def normalize_history_name(name: str) -> str:
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


def new_session_name() -> str:
    """Вернуть метку времени для имени файла истории текущей сессии.

    Returns:
        Строка вида ``YYYYMMDD_HHMMSS`` (timestamp).
    """
    return str(round(datetime.now().timestamp()))


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
    """Сохранить текущий контекст агента в файл истории (JSON pretty).

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
        filename = normalize_history_name(name)
    else:
        filename = new_session_name() + ".json"
    path = HISTORY_DIR / filename
    path.write_text(agent.context.to_json(indent=2), encoding="utf-8")
    logger.debug("История сохранена: %s (%d сообщений)", path, len(agent.context))
    return path


def autosave_history(agent: Agent, name: Optional[str] = None) -> Optional[Path]:
    """Сохранить историю сессии, не прерывая работу при ошибке.

    Используется для автоматического сохранения после каждого ответа
    модели: сбой записи не должен ронять диалог.

    Args:
        agent: Агент, чей контекст сохраняется.
        name: Имя истории (метка времени сессии); ``None`` — новая
            метка времени.

    Returns:
        Путь к файлу истории или ``None``, если запись не удалась.
    """
    try:
        return save_history(agent, name=name)
    except (OSError, ValueError) as exc:
        logger.warning("Не удалось сохранить историю: %s", exc)
        return None


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
    filename = normalize_history_name(name)
    path = HISTORY_DIR / filename
    if not path.is_file():
        raise FileNotFoundError(f"История {filename!r} не найдена в {HISTORY_DIR}")
    try:
        agent.context = Context.from_json(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise ValueError(f"Файл истории {filename!r} повреждён: {exc}") from exc
    logger.info("История загружена: %s (%d сообщений)", path, len(agent.context))
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
        "--load",
        dest="history",
        metavar="ИМЯ",
        help=(
            "загрузить сохранённую историю из ~/.baylang/history/<имя>.json "
            "перед стартом (алиас: --load; по умолчанию — новая сессия "
            "с чистой историей)"
        ),
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
    agent = Agent(
        provider=provider,
        tools=registry,
        max_iters=args.max_iters,
    )
    agent.context.add_message(TextMessage.system(system_prompt))
    logger.debug("Агент собран: модель=%s, max_iters=%d, temperature=%s",
                 model, args.max_iters, args.temperature)
    return agent


# ---------------------------------------------------------------------------
# Режимы работы
# ---------------------------------------------------------------------------


def usage_suffix(agent: Agent) -> str:
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


async def run_interactive(
    agent: Agent,
    args: argparse.Namespace,
    history_name: Optional[str] = None,
) -> int:
    """Интерактивный режим: цикл чтения строк со стандартного ввода.

    Команды: ``help``, ``clear``, ``model <имя>``, ``save [имя]``,
    ``load <имя>``, ``histories``, ``exit``/``quit``. Ошибки API не роняют
    приложение: сообщение выводится, диалог продолжается. После каждого
    ответа модели история сессии автоматически сохраняется в JSON pretty
    с именем ``<history_name>.json`` (метка времени сессии). Весь ход
    диалога дублируется в журнал ``~/.baylang/baylang.log``.

    Args:
        agent: Агент.
        args: Аргументы командной строки.
        history_name: Метка времени файла истории сессии; ``None`` —
            будет создана при первом автосохранении.

    Returns:
        Код выхода (0 — корректное завершение).
    """
    if history_name is None:
        history_name = new_session_name()
    logger.info("Сессия: %s", history_name)
    print(
        f"{APP_NAME} {APP_VERSION}. Справка: help; история: histories/save/load; "
        "очистка контекста: clear; выход: exit."
    )
    while True:
        try:
            line = input("you> ")
        except (EOFError, KeyboardInterrupt):
            print("\nДо встречи! 👋")
            return EXIT_OK

        line = line.strip()
        if not line:
            continue

        logger.debug("you> %s", line)
        parts = line.split(maxsplit=1)
        command = parts[0].lower()
        rest = parts[1].strip() if len(parts) > 1 else ""

        if command in ("exit", "quit"):
            logger.info("Завершение по команде пользователя, сессия: %s", history_name)
            print("До встречи! 👋")
            return EXIT_OK
        if command == "help":
            print(HELP_TEXT)
            continue
        if command == "clear":
            agent.reset()
            logger.info("Контекст диалога очищен")
            print("Контекст диалога очищен.")
            continue
        if command == "model":
            if not rest:
                print("Использование: model <имя модели>")
                continue
            agent.provider.model_name = rest
            logger.info("Модель изменена: %s", rest)
            print(f"Модель изменена: {rest}")
            continue
        if command == "save":
            try:
                path = save_history(agent, rest or None)
            except ValueError as exc:
                logger.warning("Ошибка сохранения истории: %s", exc)
                print(f"Ошибка: {exc}")
                continue
            except OSError as exc:
                logger.warning("Не удалось сохранить историю: %s", exc)
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
                logger.warning("Не удалось загрузить историю: %s", exc)
                print(f"Не удалось загрузить историю: {exc}")
                continue
            print(f"История загружена: {count} сообщений.")
            continue
        if command == "histories" or command == "list":
            print_histories()
            continue

        started = time.perf_counter()
        agent.context.add_message(TextMessage.user(line))
        try:
            text = await agent.send()
        except ProviderError as exc:
            logger.error("Ошибка запроса к API: %s", exc)
            print(f"assistant> Ошибка запроса: {exc}")
            print("Проверьте сеть и API-ключ, затем попробуйте ещё раз.")
            continue
        except BayLangError as exc:
            logger.error("Ошибка агента: %s", exc)
            print(f"assistant> Ошибка: {exc}")
            continue
        elapsed = time.perf_counter() - started
        logger.debug(
            "assistant> %s (%.2f с%s)", text, elapsed, usage_suffix(agent)
        )
        print(f"assistant> {text}")
        if args.verbose:
            print(
                f"[verbose] время ответа: {elapsed:.2f} с{usage_suffix(agent)}",
                file=sys.stderr,
            )
        autosave_history(agent, history_name)


async def run_app(args: argparse.Namespace) -> int:
    """Выполнить приложение: сборка агента, режим работы, автосохранение.

    По умолчанию стартует новая сессия; старая история загружается только
    по ``--history`` (алиас ``--load``) или команде ``load``. История
    сохраняется всегда: автоматически после каждого ответа и ещё раз в
    блоке ``finally`` при завершении (имя файла — метка времени сессии,
    формат JSON pretty). Ключевые события пишутся в журнал
    ``~/.baylang/baylang.log``.

    Args:
        args: Аргументы командной строки.

    Returns:
        Код выхода процесса: ``0`` — успех, ``1`` — ошибка выполнения,
        ``2`` — ошибка конфигурации.
    """
    try:
        agent = build_agent(args)
    except ConfigurationError as exc:
        logger.error("Ошибка конфигурации: %s", exc)
        print(f"Ошибка конфигурации: {exc}", file=sys.stderr)
        return EXIT_CONFIG_ERROR
    except BayLangError as exc:
        logger.error("Ошибка сборки агента: %s", exc)
        print(f"Ошибка: {exc}", file=sys.stderr)
        return EXIT_CONFIG_ERROR

    session_name = new_session_name()

    if args.history:
        try:
            count = load_history(agent, args.history)
        except (FileNotFoundError, ValueError) as exc:
            logger.error("Ошибка загрузки истории %r: %s", args.history, exc)
            print(f"Ошибка загрузки истории: {exc}", file=sys.stderr)
            return EXIT_CONFIG_ERROR
        print(
            f"Загружена история '{args.history}': {count} сообщений.",
            file=sys.stderr,
        )

    if args.verbose:
        api_key = os.environ.get(ENV_API_KEY, "")
        logger.debug(
            "Модель: %s; ключ API: %s; промпт: %s; сессия: %s",
            agent.provider.get_model_name(),
            mask_api_key(api_key),
            DEFAULT_PROMPT_FILE,
            session_name,
        )

    exit_code = EXIT_OK
    try:
        exit_code = await run_interactive(agent, args, history_name=session_name)
    except KeyboardInterrupt:
        print("\nДо встречи! 👋")
        exit_code = EXIT_OK
    except ProviderError as exc:
        logger.error("Ошибка выполнения: %s", exc)
        print(f"Ошибка выполнения: {exc}", file=sys.stderr)
        exit_code = EXIT_RUNTIME_ERROR
    except BayLangError as exc:
        logger.error("Ошибка: %s", exc)
        print(f"Ошибка: {exc}", file=sys.stderr)
        exit_code = EXIT_RUNTIME_ERROR
    finally:
        # История сохраняется всегда — при любом способе завершения.
        try:
            save_history(agent, name=session_name)
        except (OSError, ValueError) as exc:
            logger.warning("Не удалось сохранить историю: %s", exc)
            print(f"Не удалось сохранить историю: {exc}", file=sys.stderr)
        await agent.disconnect()
    return exit_code


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Оркестратор приложения: парсинг аргументов и запуск async-ядра.

    Новая сессия стартует по умолчанию; история диалога сохраняется
    автоматически и всегда (имя файла — метка времени, JSON pretty).
    Журнал работы пишется в ``~/.baylang/baylang.log`` (ротация 1 МБ × 3,
    в файл попадают сообщения начиная с INFO).

    Args:
        argv: Аргументы командной строки; ``None`` — ``sys.argv[1:]``.

    Returns:
        Код выхода процесса: ``0`` — успех, ``1`` — ошибка выполнения,
        ``2`` — ошибка конфигурации.
    """
    args = parse_args(argv)
    ensure_app_dirs()
    setup_logging(verbose=args.verbose)
    logger.info(
        "%s %s запущен (verbose=%s, аргументы: %s)",
        APP_NAME,
        APP_VERSION,
        args.verbose,
        vars(args),
    )

    if args.list_histories:
        print_histories()
        return EXIT_OK

    try:
        return asyncio.run(run_app(args))
    except KeyboardInterrupt:
        print("\nДо встречи! 👋")
        return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
