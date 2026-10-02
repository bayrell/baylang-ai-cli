"""Общий каркас файловых инструментов (tools) BayLang AI.

Модуль содержит абстрактный базовый класс :class:`FsTool`, песочницу
рабочей директории и вспомогательные функции, общие для всех файловых
инструментов ``fs_*``.

:class:`FsTool` **наследует** :class:`~ai.Tool` и реализует его контракт
(``name`` / ``description`` / ``parameters_schema`` / ``hint`` /
:meth:`~ai.Tool.get_schema` / :meth:`~ai.Tool.execute`), поэтому экземпляр
конкретного инструмента регистрируется в :class:`~ai.ToolRegistry` напрямую,
без дополнительной обёртки ``Tool(...)``. Атрибуты ``name``, ``description``,
``parameters_schema`` и ``hint`` заполняются из классовых :attr:`FsTool.NAME`,
:attr:`FsTool.DESCRIPTION`, :attr:`FsTool.PARAMETERS` и :attr:`FsTool.HINT`
при инициализации.

Изоляция файловой системы. Инструменты работают **только** внутри рабочей
директории (``root`` — текущая папка процесса). Любой пользовательский путь
проходит через :meth:`FsTool.resolve_path`, который:

* разрешает ``.`` / ``..`` и символические ссылки (``Path.resolve``);
* отклоняет выход за пределы ``root`` (атаки вида ``../../etc/passwd``);
* запрещает доступ к файлам из :attr:`FsTool.DENIED_NAMES` (например,
  ``.env`` с секретами — API-ключ не должен попадать в вывод);
* защищает сам корень от удаления и переименования (``allow_root=False``).

При обходе деревьев служебные каталоги (:attr:`FsTool.SKIP_DIR_NAMES`:
``.git``, ``__pycache__`` и т.п.) пропускаются, чтобы поиск не тратил
время и не засорял результат.

Контракт инструмента. Каждый инструмент — наследник :class:`FsTool`
в отдельном файле (один файл — один класс) и реализует:

* :attr:`FsTool.HINT` — подсказка для системного промпта (собирается
  реестром через :meth:`~ai.ToolRegistry.build_hint`);
* :meth:`FsTool.run` — асинхронное выполнение операции; возвращает
  структурированный словарь ``{"tool": имя, "status": "ok"/"error", ...}``;
* :meth:`FsTool.format_message` — метод форматирования результата для вывода
  пользователю через ``format_display``.

Ошибки (включая попытку выхода за песочницу) наружу не бросаются, а
возвращаются в виде ``{"status": "error", "error": ...}``: модель получает
аккуратный результат, а вывод — красивое сообщение через ``format_message``.
"""

from __future__ import annotations

import abc
from pathlib import Path
from typing import Any, ClassVar, Iterator, Optional, Union

from ..ai import Tool  # type: ignore[no-redef]

__all__ = [
    "FsTool",
    "FsToolError",
    "human_size",
    "MAX_FILE_SIZE",
    "MAX_LIST_ENTRIES",
    "MAX_TEXT_SIZE",
    "SANDBOX_NOTE",
]

#: Максимальный размер читаемого/редактируемого файла (1 МБ).
MAX_FILE_SIZE = 1024 * 1024

#: Максимальный размер записываемого текста (1 МБ).
MAX_TEXT_SIZE = 1024 * 1024

#: Максимум записей в одном результате (списки, поиск, совпадения).
MAX_LIST_ENTRIES = 500


class FsToolError(Exception):
    """Внутренняя ошибка файлового инструмента.

    Используется для отказов песочницы и ошибок валидации; перехватывается
    в :meth:`FsTool.execute` и превращается в ``{"status": "error"}``.
    """


def human_size(size: int) -> str:
    """Представить размер файла в человекочитаемом виде.

    Args:
        size: Размер в байтах.

    Returns:
        Строка вида ``512 Б``, ``12.3 КБ`` или ``1.5 МБ``; для
        отрицательных размеров — ``?``.
    """
    if size < 0:
        return "?"
    if size < 1024:
        return f"{size} Б"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f} КБ"
    return f"{size / (1024 * 1024):.1f} МБ"


class FsTool(Tool, abc.ABC):
    """Базовый класс файлового инструмента с песочницей рабочей директории.

    Наследует :class:`~ai.Tool`: имя, описание, JSON-схема параметров
    и подсказка системного промпта берутся из классовых :attr:`NAME`,
    :attr:`DESCRIPTION`, :attr:`PARAMETERS` и :attr:`HINT`, а обработчиком
    инструмента служит его :meth:`run`. Благодаря этому экземпляр наследника
    регистрируется в :class:`~ai.ToolRegistry` напрямую
    (``registry.register(instance)``).

    Наследник реализует :attr:`HINT`, :meth:`run` и :meth:`format_message`.

    Args:
        root: Корень песочницы (рабочая директория); ``None`` — текущая
            папка процесса.

    Raises:
        FsToolError: Если ``root`` не существует или не является папкой.
        ConfigurationError: Если :attr:`NAME` или :attr:`DESCRIPTION`
            пусты (проверка :class:`~ai.Tool`).
    """

    #: Уникальное имя инструмента (латиницей, без пробелов).
    NAME: ClassVar[str] = ""

    #: Описание для модели.
    DESCRIPTION: ClassVar[str] = ""

    #: JSON-схема параметров в формате OpenAI Function.
    PARAMETERS: ClassVar[dict[str, Any]] = {"type": "object", "properties": {}}

    #: Подсказка для системного промпта (короткая строка о сути инструмента).
    HINT: ClassVar[str] = ""

    #: Имена файлов/папок, полностью закрытых от инструментов (секреты).
    DENIED_NAMES: ClassVar[frozenset[str]] = frozenset({".env"})

    #: Служебные папки, пропускаемые при обходе деревьев.
    SKIP_DIR_NAMES: ClassVar[frozenset[str]] = frozenset(
        {".git", "__pycache__", "node_modules", ".venv", ".idea"}
    )

    def __init__(self, root: Optional[Union[str, Path]] = None) -> None:
        base = Path(root) if root is not None else Path.cwd()
        self.root = base.resolve()
        if not self.root.is_dir():
            raise FsToolError(
                f"Рабочая директория {self.root} не существует или не является папкой"
            )
        super().__init__(
            name=self.NAME,
            description=self.DESCRIPTION,
            parameters_schema=dict(self.PARAMETERS),
            handler=self.run,
            hint=self.HINT,
        )

    async def execute(  # type: ignore[override]
        self,
        params: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Выполнить инструмент, переведя любую ошибку в формат результата.

        Точка входа :class:`~ai.Tool`: :class:`~ai.ToolRegistry` вызывает
        её с позиционным словарём аргументов из ответа модели. Для
        удобства поддерживаются и именованные аргументы
        (``execute(path=".")``) — они сливаются со словарём ``params``.

        Args:
            params: Параметры вызова (по JSON-схеме ``parameters_schema``).
            kwargs: Дополнительные именованные параметры вызова.

        Returns:
            Словарь ``{"tool": ..., "status": "ok"/"error", ...}`` —
            исключений наружу не выпускает.
        """
        if params is not None and not isinstance(params, dict):
            return self.error("Параметры инструмента должны быть словарём")
        arguments: dict[str, Any] = dict(params or {})
        arguments.update(kwargs)
        try:
            result = await self.run(**arguments)
        except FsToolError as exc:
            return self.error(str(exc))
        except TypeError as exc:
            return self.error(f"Некорректные параметры вызова: {exc}")
        except OSError as exc:
            return self.error(f"Файловая ошибка: {exc}")
        return result

    @abc.abstractmethod
    async def run(self, **params: Any) -> dict[str, Any]:
        """Выполнить операцию инструмента.

        Args:
            params: Параметры вызова по JSON-схеме.

        Returns:
            Словарь результата (:meth:`ok`).

        Raises:
            FsToolError: При отказе песочницы или ошибке валидации.
        """

    @staticmethod
    @abc.abstractmethod
    def format_message(params: dict[str, Any], result: Any = None) -> str:
        """Отформатировать результат инструмента для ``format_display``.

        Args:
            data: Словарь результата (:meth:`ok` или :meth:`error`).

        Returns:
            Строка, готовая к печати пользователю.
        """

    def ok(self, **fields: Any) -> dict[str, Any]:
        """Собрать успешный результат с меткой инструмента.

        Args:
            fields: Поля результата (путь, записи, совпадения и т.п.).

        Returns:
            Словарь ``{"tool": имя, "status": "ok", ...}``.
        """
        return fields
    
    def error(self, message: str) -> str:
        return {"error": message}
    
    def rel(self, path: Path) -> str:
        """Вернуть путь относительно корня песочницы для вывода.

        Args:
            path: Абсолютный путь внутри ``root``.

        Returns:
            Относительный путь; для самого корня — ``"."``.
        """
        try:
            relative = path.resolve().relative_to(self.root)
        except ValueError:
            return str(path)
        text = str(relative)
        return text if text else "."

    def is_denied(self, path: Path) -> bool:
        """Проверить, закрыт ли путь инструментами (секреты вроде ``.env``).

        Args:
            path: Проверяемый путь.

        Returns:
            ``True``, если любой компонент пути входит в :attr:`DENIED_NAMES`.
        """
        return any(part in self.DENIED_NAMES for part in Path(path).parts)

    def resolve_path(
        self,
        path: Union[str, Path],
        *,
        allow_root: bool = False,
        must_exist: bool = False,
    ) -> Path:
        """Разрешить пользовательский путь внутри песочницы.

        Args:
            path: Путь относительно корня (или абсолютный внутри корня).
            allow_root: Разрешить ссылаться на саму рабочую директорию
                (нужно для list/find/grep; для write/delete/rename — нет).
            must_exist: Требовать, чтобы путь существовал.

        Returns:
            Абсолютный разрешённый путь внутри ``root``.

        Raises:
            FsToolError: Если путь пуст, выходит за пределы ``root``,
                указывает на закрытый файл (``.env``), на сам корень
                без ``allow_root`` или не существует при ``must_exist``.
        """
        raw = str(path).strip() if path is not None else ""
        if not raw:
            raise FsToolError("Путь не может быть пустым")
        if "~" in raw:
            raise FsToolError(
                f"Путь {raw!r} выходит за пределы рабочей директории: домашняя папка (~) недоступна"
            )
        target = (self.root / Path(raw)).resolve()
        if target != self.root and not target.is_relative_to(self.root):
            raise FsToolError(
                f"Путь {raw!r} выходит за пределы рабочей директории {self.root} — доступ запрещён"
            )
        if self.is_denied(target):
            raise FsToolError(
                f"Доступ к {self.rel(target)!r} запрещён: файлы {sorted(self.DENIED_NAMES)} закрыты инструментами"
            )
        if target == self.root and not allow_root:
            raise FsToolError(
                "Операция над самой рабочей директорией запрещена — укажите файл или папку внутри неё"
            )
        if must_exist and not target.exists():
            raise FsToolError(f"{self.rel(target)!r} не найдено в рабочей директории")
        return target

    def iter_tree(
        self,
        start: Path,
        recursive: bool = True,
        show_hidden: bool = False,
    ) -> Iterator[tuple[Path, bool]]:
        """Обойти содержимое папки, пропуская служебные и закрытые записи.

        Args:
            start: Начальная папка (внутри песочницы).
            recursive: Спускаться в подпапки.
            show_hidden: Показывать имена, начинающиеся с ``.``.

        Yields:
            Кортежи ``(путь, это папка)`` в алфавитном порядке имён.
        """
        stack: list[Path] = [start]
        while stack:
            current = stack.pop()
            try:
                entries = sorted(current.iterdir(), key=lambda p: p.name)
            except OSError:
                continue
            for entry in entries:
                if self.is_denied(entry):
                    continue
                if not show_hidden and entry.name.startswith("."):
                    continue
                try:
                    is_dir = entry.is_dir()
                except OSError:
                    continue
                yield entry, is_dir
                if recursive and is_dir and entry.name not in self.SKIP_DIR_NAMES:
                    stack.append(entry)

    def read_text(self, path: Path) -> str:
        """Прочитать текстовый файл (UTF-8) с проверкой размера.

        Args:
            path: Путь к файлу внутри песочницы.

        Returns:
            Текст файла.

        Raises:
            FsToolError: Если файл не является обычным, превышает
                :data:`MAX_FILE_SIZE` или не декодируется как UTF-8.
        """
        if not path.is_file():
            raise FsToolError(f"{self.rel(path)!r} не является файлом")
        size = path.stat().st_size
        if size > MAX_FILE_SIZE:
            raise FsToolError(
                f"{self.rel(path)!r} слишком большой для чтения "
                f"({human_size(size)}, лимит {human_size(MAX_FILE_SIZE)})"
            )
        try:
            return path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise FsToolError(
                f"{self.rel(path)!r} не является текстовым файлом UTF-8"
            ) from exc

    def __repr__(self) -> str:
        return f"{type(self).__name__}(root={str(self.root)!r})"
