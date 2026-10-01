#!/usr/bin/env python3

"""Точка входа BayLang AI: интерактивный чат и одноразовый запрос.

Запуск из корня репозитория::

    python -m src.main --prompt "Скажи привет"
    python -m src                # интерактивный режим (через src/__main__.py)
    python src/main.py           # прямой запуск файла (fallback-импорт)

Конфигурация (хранится в переменных окружения):

* ключ API — ``OPENROUTER_API_KEY`` (обязательна);
* модель — ``OPENROUTER_MODEL`` или аргумент ``--model``
  (приоритет: аргумент > переменная окружения > значение по умолчанию);
* число последних сообщений, показываемых при загрузке истории, —
  ``BAYLANG_SHOW_MESSAGES`` или аргумент ``--show-messages``.

Домашняя папка приложения (``~/.baylang``):

* ``prompt.txt`` — системный промпт (создаётся с промптом по умолчанию,
  если файла нет; путь можно переопределить через ``--prompt-file``);
* ``history/`` — сохранённые истории диалогов (JSON pretty, имя файла —
  метка времени сессии, например ``20250101_120000.json``);
* ``logs/`` — логи работы приложения (``baylang.log`` с ротацией:
  до 5 файлов-бэкапов по 1 МБ).

По умолчанию всегда запускается **новая сессия**; сохранённую историю
можно подхватить флагом ``--history <имя>`` или командой ``load <имя>``
в интерактивном режиме. При загрузке истории пользователю показываются
её последние сообщения (количество настраивается). История сохраняется
автоматически и всегда: после каждого ответа модели и ещё раз при
завершении работы. Список последних историй выводится командой
``histories`` или флагом ``--list-histories``.

Агенту регистрируются **файловые инструменты** (пакет ``src/tools``):
``fs_list``, ``fs_find``, ``fs_search_regex``, ``fs_create``, ``fs_edit``,
``fs_rename``, ``fs_delete``. Они ограничены рабочей директорией запуска
(песочницей) — доступ ко всей файловой системе модель не получает.
Подсказка системного промпта об инструментах собирается из их ``HINT``
через :meth:`ToolRegistry.build_hint`.

Все сообщения диалога (текст модели, вызовы инструментов, результаты их
работы) выводятся через единую функцию форматирования :func:`format_display`;
результаты инструментов ``fs_*`` дополнительно форматируются собственным
``format_message`` каждого класса (через ``format_tool_payload``).
Интерактивный режим получает сообщения по мере появления из асинхронного
генератора :meth:`Agent.send_with`.

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
from dotenv import load_dotenv
from datetime import datetime
from logging.handlers import RotatingFileHandler
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
    ToolResultMessage,
    ToolsMessage,
)
from tools import (  # type: ignore[no-redef]
    format_tool_payload,
    register_fs_tools,
)

__all__ = [
    "build_agent",
    "format_display",
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
    "print_last_messages",
]

load_dotenv()

APP_NAME = "BayLang AI"
APP_VERSION = "1.0.0"
DEFAULT_MODEL = "openrouter/auto"
ENV_API_KEY = "OPENROUTER_API_KEY"
ENV_MODEL = "OPENROUTER_MODEL"
ENV_SHOW_MESSAGES = "BAYLANG_SHOW_MESSAGES"

EXIT_OK = 0
EXIT_RUNTIME_ERROR = 1
EXIT_CONFIG_ERROR = 2

# Домашняя папка приложения: промпт, истории диалогов и логи.
APP_DIR = Path.home() / ".baylang"
DEFAULT_PROMPT_FILE = APP_DIR / "prompt.txt"
HISTORY_DIR = APP_DIR / "history"
HISTORY_LIST_LIMIT = 10

# Сколько последних сообщений показывать при загрузке истории.
# Переопределяется аргументом --show-messages или переменной
# окружения BAYLANG_SHOW_MESSAGES (приоритет: аргумент > env > default).
DEFAULT_SHOW_MESSAGES = 10

# Логи работы приложения: ротация, чтобы файл не разрастался бесконечно.
LOG_DIR = APP_DIR / "logs"
LOG_FILE = LOG_DIR / "baylang.log"
LOG_MAX_BYTES = 1024 * 1024  # 1 МБ на файл
LOG_BACKUP_COUNT = 5  # количество файлов-бэкапов (baylang.log.1, ...)

SYSTEM_PROMPT = (
    "Ты — полезный ассистент BayLang AI. Отвечай кратко и по делу, "
    "по-русски, если пользователь не просит иначе."
)

HELP_TEXT = """\
Команды:
  help                — показать эту справку
  clear               — очистить контекст диалога
  model <имя>         — сменить модель на лету (например: model openrouter/auto)
  save [имя]          — сохранить историю под другим именем (по умолчанию — метка времени)
  load <имя>          — загрузить историю из ~/.baylang/history
  histories           — список последних историй
  exit, quit          — завершить работу (также Ctrl+C и Ctrl+D)
Любая другая строка отправляется модели как запрос.
История сессии сохраняется автоматически после каждого ответа и при выходе.
Файл промпта: ~/.baylang/prompt.txt; история: ~/.baylang/history/;
логи: ~/.baylang/logs/baylang.log."""

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Форматирование вывода
# ---------------------------------------------------------------------------

ROLE_LABELS = {
    TextMessage.ROLE_USER: "you",
    TextMessage.ROLE_AI: "assistant",
    TextMessage.ROLE_SYSTEM: "system",
    TextMessage.ROLE_TOOL: "tool",
}

def format_display(message: TextMessage) -> str:
    """Отформатировать сообщение диалога для вывода пользователю.

    Единая точка форматирования всех сообщений: текст модели, реплики
    пользователя, системные сообщения, вызовы инструментов и результаты
    их работы. Результат инструмента показывается как ``tool[<id>] > ...``,
    вызовы инструментов — многострочно с аргументами и идентификаторами,
    остальные сообщения — с меткой роли (``you``, ``assistant``, ``system``).

    Для результатов файловых инструментов (``fs_*``) используется их
    собственный ``format_message``: JSON с полем ``tool`` распознаётся
    через ``format_tool_payload`` и превращается в человекочитаемый вид.
    Если распознать результат не удалось, выводится сырое содержимое.

    Args:
        message: Сообщение диалога (:class:`TextMessage` или его наследник
            :class:`ToolResultMessage` / :class:`ToolsMessage`).

    Returns:
        Строка, готовая к печати.
    """
    if isinstance(message, ToolResultMessage):
        custom = format_tool_payload(message.content)
        if custom:
            return f"tool[{message.tool_id}]> {custom}"
        content = message.content.strip()
        if not content:
            return f"tool[{message.tool_id}]> (пустой результат)"
        return f"tool[{message.tool_id}]> {content}"

    if isinstance(message, ToolsMessage):
        lines: list[str] = []
        content = message.content.strip()
        if content:
            lines.append(f"assistant> {content}")
        items: list[str] = []
        for index, tool in enumerate(message.tools or [], start=1):
            if not isinstance(tool, dict):
                continue
            function = tool.get("function") or {}
            name = function.get("name") or "?"
            arguments = function.get("arguments") or "{}"
            tool_id = tool.get("id") or f"tool_{index}"
            items.append(f"  ⚙ {name}({arguments}) — id={tool_id}")
        if items:
            lines.append("assistant> вызывает инструменты:")
            lines.extend(items)
        if not lines:
            return "assistant> (вызовы инструментов без данных)"
        return "\n".join(lines)

    label = ROLE_LABELS.get(message.role, message.role)
    content = message.content.strip()
    if not content:
        return f"{label}> (пустое сообщение)"
    return f"{label}> {content}"


# ---------------------------------------------------------------------------
# Домашняя папка, промпт и истории
# ---------------------------------------------------------------------------


def ensure_app_dirs() -> None:
    """Создать домашнюю папку приложения, каталоги историй и логов.

    Если файл ``prompt.txt`` отсутствует, записывается промпт по умолчанию,
    чтобы пользователь мог отредактировать его перед запуском.
    """
    APP_DIR.mkdir(parents=True, exist_ok=True)
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if not DEFAULT_PROMPT_FILE.exists():
        try:
            DEFAULT_PROMPT_FILE.write_text(SYSTEM_PROMPT + "\n", encoding="utf-8")
        except OSError as exc:
            logger.warning("Не удалось создать %s: %s", DEFAULT_PROMPT_FILE, exc)


def setup_logging(verbose: bool = False) -> None:
    """Настроить логирование: консоль + файл в ``~/.baylang/logs/baylang.log``.

    Уровень логирования: ``DEBUG`` при ``--verbose``, иначе ``INFO`` для
    файлового обработчика и ``WARNING`` для консоли. Файл лога ротируется:
    при превышении ``LOG_MAX_BYTES`` создаётся бэкап (всего до
    ``LOG_BACKUP_COUNT`` штук), поэтому логи не занимают много места.

    Args:
        verbose: Включить подробное (DEBUG) логирование.
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # Не дублируем обработчики при повторном вызове (например, в тестах).
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()

    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG if verbose else logging.WARNING)
    console.setFormatter(formatter)
    root.addHandler(console)

    try:
        file_handler = RotatingFileHandler(
            LOG_FILE,
            maxBytes=LOG_MAX_BYTES,
            backupCount=LOG_BACKUP_COUNT,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
    except OSError as exc:
        # Лог-файл не критичен для работы: продолжаем только с консолью.
        console.setLevel(logging.WARNING)
        logging.getLogger(__name__).warning(
            "Не удалось открыть файл лога %s: %s", LOG_FILE, exc
        )


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


def new_session_name() -> str:
    """Вернуть метку времени для имени файла истории текущей сессии.

    Returns:
        Строка вида timestamp.
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
        name: Имя истории; ``None`` — метка времени в формате timestamp.

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
        filename = new_session_name() + ".json"
    path = HISTORY_DIR / filename
    path.write_text(agent.context.to_json(indent=2), encoding="utf-8")
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
    filename = _normalize_history_name(name)
    path = HISTORY_DIR / filename
    if not path.is_file():
        raise FileNotFoundError(f"История {filename!r} не найдена в {HISTORY_DIR}")
    try:
        agent.context = Context.from_json(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise ValueError(f"Файл истории {filename!r} повреждён: {exc}") from exc
    return len(agent.context)


def print_last_messages(context: Context, count: Optional[int] = None) -> None:
    """Показать последние сообщения загруженного контекста.

    Использует общую функцию форматирования :func:`format_display`:
    текст модели, вызовы инструментов и результаты их работы выводятся
    одинаково — так пользователь сразу видит, чем закончилась предыдущая
    часть диалога.

    Args:
        context: Диалоговый контекст (обычно — только что загруженный).
        count: Сколько последних сообщений показать; ``None`` — значение
            по умолчанию (:data:`DEFAULT_SHOW_MESSAGES`).
    """
    if count is None:
        count = DEFAULT_SHOW_MESSAGES
    messages = context.last_messages(count)
    total = len(context)
    if not messages:
        print("(контекст пуст)")
        return
    for message in messages:
        if isinstance(message, ToolResultMessage):
            continue
        if isinstance(message, TextMessage) and message.role == TextMessage.ROLE_SYSTEM:
            continue
        print(format_display(message))


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
        "--model",
        help=f"модель OpenRouter (по умолчанию: {DEFAULT_MODEL}, env {ENV_MODEL})",
    )
    parser.add_argument(
        "--history",
        "--load",
        metavar="ИМЯ",
        help=(
            "загрузить сохранённую историю из ~/.baylang/history/<имя>.json "
            "перед стартом (по умолчанию — новая сессия с чистой историей)"
        ),
    )
    parser.add_argument(
        "--show-messages",
        type=int,
        default=None,
        metavar="N",
        help=(
            "сколько последних сообщений показывать при загрузке истории "
            f"(по умолчанию: {DEFAULT_SHOW_MESSAGES}, env {ENV_SHOW_MESSAGES})"
        ),
    )
    parser.add_argument(
        "--list-histories",
        "--list",
        action="store_true",
        help="вывести список последних историй и завершиться",
    )
    parser.add_argument(
        "--max-iters",
        type=int,
        default=100,
        help="максимум итераций tool (по умолчанию: 100)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="температура генерации (по умолчанию: 0.7)",
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


def get_show_messages(args: argparse.Namespace) -> int:
    """Определить число последних сообщений для показа при загрузке истории.

    Приоритет: аргумент ``--show-messages`` > переменная окружения
    ``BAYLANG_SHOW_MESSAGES`` > значение по умолчанию.

    Args:
        args: Аргументы командной строки (:func:`parse_args`).

    Returns:
        Положительное число сообщений (некорректные значения заменяются
        значением по умолчанию).
    """
    value = args.show_messages
    if value is None:
        env_value = os.environ.get(ENV_SHOW_MESSAGES, "").strip()
        if env_value:
            try:
                value = int(env_value)
            except ValueError:
                logger.warning(
                    "Некорректное значение %s=%r, используется %d",
                    ENV_SHOW_MESSAGES,
                    env_value,
                    DEFAULT_SHOW_MESSAGES,
                )
                value = None
    if value is None:
        return DEFAULT_SHOW_MESSAGES
    if value < 1:
        logger.warning(
            "Число сообщений для показа должно быть >= 1, получено %d; "
            "используется %d",
            value,
            DEFAULT_SHOW_MESSAGES,
        )
        return DEFAULT_SHOW_MESSAGES
    return value


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

    Агенту регистрируются файловые инструменты ``fs_*`` (пакет ``src/tools``)
    с корнем-песочницей — текущей рабочей директорией запуска. Модель
    получает доступ только к файлам внутри неё; в системный промпт
    добавляется подсказка об инструментах, собираемая из их ``HINT``
    через :meth:`ToolRegistry.build_hint` (заголовок
    :data:`tools.FS_TOOLS_INTRO`, правило песочницы
    :data:`tools.SANDBOX_NOTE`).

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
    registered_tools = register_fs_tools(registry, root=Path.cwd())
    tools_hint = registry.build_hint()
    agent = Agent(
        provider=provider,
        tools=registry,
        max_iters=args.max_iters,
    )
    agent.context.add_message(
        TextMessage.system(f"{system_prompt}\n\n{tools_hint}")
    )
    logger.info(
        "Агент собран: модель %s, ключ API %s; инструменты: %s",
        model,
        mask_api_key(api_key),
        ", ".join(registered_tools) or "нет",
    )
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
    """Цикл чтения строк со стандартного ввода.

    Команды: ``help``, ``clear``, ``model <имя>``, ``save [имя]``,
    ``load <имя>``, ``histories``, ``exit``/``quit``. Ошибки API не роняют
    приложение: сообщение выводится, диалог продолжается. После каждого
    ответа модели история сессии автоматически сохраняется в JSON pretty
    с именем ``<history_name>.json`` (метка времени сессии).

    Сообщения выводятся через единую функцию форматирования
    :func:`format_display`: текст модели, вызовы инструментов и результаты
    их выполнения приходят из асинхронного генератора :meth:`Agent.send_with`
    и печатаются по мере появления. При загрузке истории показываются её
    последние сообщения (количество — из настроек ``--show-messages`` /
    ``BAYLANG_SHOW_MESSAGES``).

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
    show_messages = get_show_messages(args)
    logger.info("Сессия %s (показ сообщений при загрузке: %d)", history_name, show_messages)
    print(
        f"{APP_NAME} {APP_VERSION}. Справка: help; история: histories/save/load; "
        "очистка контекста: clear; выход: exit."
    )
    while True:
        try:
            line = input("you> ")
        except (EOFError, KeyboardInterrupt):
            print("\nДо встречи! 👋")
            logger.info("Сессия %s завершена (EOF/KeyboardInterrupt)", history_name)
            return EXIT_OK

        line = line.strip()
        if not line:
            continue

        parts = line.split(maxsplit=1)
        command = parts[0].lower()
        rest = parts[1].strip() if len(parts) > 1 else ""

        if command in ("exit", "quit"):
            print("До встречи! 👋")
            logger.info("Сессия %s завершена по команде %s", history_name, command)
            return EXIT_OK
        if command == "help":
            print(HELP_TEXT)
            continue
        if command == "clear":
            agent.reset()
            print("Контекст диалога очищен.")
            logger.info("Контекст диалога очищен (сессия %s)", history_name)
            continue
        if command == "model":
            if not rest:
                print("Использование: model <имя модели>")
                continue
            old_model = agent.provider.model_name
            agent.provider.model_name = rest
            print(f"Модель изменена: {rest}")
            logger.info("Модель изменена: %s -> %s", old_model, rest)
            continue
        if command == "save":
            try:
                path = save_history(agent, rest or None)
            except ValueError as exc:
                print(f"Ошибка: {exc}")
                logger.warning("Ошибка сохранения истории: %s", exc)
                continue
            except OSError as exc:
                print(f"Не удалось сохранить историю: {exc}")
                logger.warning("Не удалось сохранить историю: {exc}")
                continue
            print(f"История сохранена: {path}")
            logger.info("История сохранена: %s", path)
            continue
        if command == "load":
            if not rest:
                print("Использование: load <имя> (список историй: histories)")
                continue
            try:
                count = load_history(agent, rest)
            except (FileNotFoundError, ValueError) as exc:
                print(f"Не удалось загрузить историю: {exc}")
                logger.warning("Не удалось загрузить историю %r: %s", rest, exc)
                continue
            print(f"История загружена: {count} сообщений.")
            print_last_messages(agent.context, show_messages)
            logger.info(
                "История %r загружена: %d сообщений (показано %d)",
                rest,
                count,
                show_messages,
            )
            continue
        if command == "histories" or command == "list":
            print_histories()
            continue

        started = time.perf_counter()
        agent.context.add_message(TextMessage.user(line))
        try:
            async for message in agent.send_with():
                print(format_display(message))
        except ProviderError as exc:
            print(f"assistant> Ошибка запроса: {exc}")
            print("Проверьте сеть и API-ключ, затем попробуйте ещё раз.")
            logger.error("Ошибка запроса к провайдеру: %s", exc)
            continue
        except BayLangError as exc:
            print(f"assistant> Ошибка: {exc}")
            logger.error("Ошибка: %s", exc)
            continue
        elapsed = time.perf_counter() - started
        logger.info(
            "Ответ модели получен за %.2f с%s",
            elapsed,
            usage_suffix(agent),
        )
        if args.verbose:
            print(
                f"[verbose] время ответа: {elapsed:.2f} с{usage_suffix(agent)}",
                file=sys.stderr,
            )
        autosave_history(agent, history_name)


async def run_app(args: argparse.Namespace) -> int:
    """Выполнить приложение: сборка агента, режим работы, автосохранение.

    По умолчанию стартует новая сессия; старая история загружается только
    по ``--history`` или командой ``load`` — при загрузке пользователю
    показываются последние сообщения истории (количество задаётся
    настройкой ``--show-messages`` / ``BAYLANG_SHOW_MESSAGES``). История
    сохраняется всегда: автоматически после каждого ответа и ещё раз в
    блоке ``finally`` при завершении (имя файла — метка времени сессии,
    формат JSON pretty).

    Args:
        args: Аргументы командной строки.

    Returns:
        Код выхода процесса: ``0`` — успех, ``1`` — ошибка выполнения,
        ``2`` — ошибка конфигурации.
    """
    try:
        agent = build_agent(args)
    except ConfigurationError as exc:
        print(f"Ошибка конфигурации: {exc}", file=sys.stderr)
        logger.error("Ошибка конфигурации: %s", exc)
        return EXIT_CONFIG_ERROR
    except BayLangError as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        logger.error("Ошибка: %s", exc)
        return EXIT_CONFIG_ERROR

    session_name = new_session_name()
    logger.info("Запуск сессии %s (модель: %s)", session_name, agent.provider.get_model_name())

    if args.history:
        try:
            count = load_history(agent, args.history)
        except (FileNotFoundError, ValueError) as exc:
            print(f"Ошибка загрузки истории: {exc}", file=sys.stderr)
            logger.error("Ошибка загрузки истории %r: %s", args.history, exc)
            return EXIT_CONFIG_ERROR
        print(
            f"Загружена история '{args.history}': {count} сообщений.",
            file=sys.stderr,
        )
        # Показать последние сообщения загруженной истории.
        print_last_messages(agent.context, get_show_messages(args))
        logger.info("Загружена история %r: %d сообщений", args.history, count)

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
        logger.info("Сессия %s прервана (KeyboardInterrupt)", session_name)
        exit_code = EXIT_OK
    except ProviderError as exc:
        print(f"Ошибка выполнения: {exc}", file=sys.stderr)
        logger.error("Ошибка выполнения: %s", exc)
        exit_code = EXIT_RUNTIME_ERROR
    except BayLangError as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        logger.error("Ошибка: %s", exc)
        exit_code = EXIT_RUNTIME_ERROR
    finally:
        await agent.disconnect()
        logger.info("Сессия %s завершена с кодом %d", session_name, exit_code)
    return exit_code


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Оркестратор приложения: парсинг аргументов и запуск async-ядра.

    Новая сессия стартует по умолчанию; история диалога сохраняется
    автоматически и всегда (имя файла — метка времени, JSON pretty).
    Логи работы пишутся в ``~/.baylang/logs/baylang.log`` (с ротацией).

    Args:
        argv: Аргументы командной строки; ``None`` — ``sys.argv[1:]``.

    Returns:
        Код выхода процесса: ``0`` — успех, ``1`` — ошибка выполнения,
        ``2`` — ошибка конфигурации.
    """
    args = parse_args(argv)
    ensure_app_dirs()
    setup_logging(verbose=args.verbose)
    logger.debug("Логирование настроено: файл %s, verbose=%s", LOG_FILE, args.verbose)

    if args.list_histories:
        print_histories()
        return EXIT_OK

    try:
        return asyncio.run(run_app(args))
    except KeyboardInterrupt:
        print("\nДо встречи! 👋")
        logger.info("Приложение прервано (KeyboardInterrupt)")
        return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
