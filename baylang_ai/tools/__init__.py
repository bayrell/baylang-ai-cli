"""Файловые инструменты (tools) BayLang AI.

Пакет содержит инструменты работы с файлами проекта — по одному классу
в файле. Все инструменты ограничены **рабочей директорией** (песочницей,
см. :mod:`tools.base`): модель не получает доступ ко всей файловой системе.

Доступные инструменты:

===================== =============== =====================================
Имя                   Файл            Назначение
===================== =============== =====================================
``fs_list``           list_dir.py     список файлов и папок
``fs_find``           find_file.py    поиск файлов по glob-шаблону имени
``fs_create``         create_file.py  создание файла
``fs_delete``         delete_file.py  удаление файла/папки
``fs_rename``         rename_file.py  переименование/перенос
``fs_edit``           edit_file.py    редактирование (замена/запись)
``fs_search_regex``   regex_search.py поиск по regexp по содержимому
===================== =============== =====================================

Каждый класс наследует :class:`~base.FsTool`, который в свою очередь
наследует :class:`~ai.Tool` — поэтому экземпляр инструмента регистрируется
в :class:`~ai.ToolRegistry` напрямую, без обёртки ``Tool(...)``:

* ``execute(params)`` — асинхронное выполнение (точка входа
  :class:`~ai.Tool`), всегда возвращает словарь ``{"tool", "status", ...}``;
* ``HINT`` — подсказка системного промпта; реестр собирает их в единую
  секцию через :meth:`~ai.ToolRegistry.build_hint` (см. :data:`FS_TOOLS_INTRO`
  и :data:`tools.base.SANDBOX_NOTE`);
* ``format_message(data)`` — статический метод форматирования результата,
  который :func:`format_tool_payload` подставляет в ``format_display``.

Регистрация в агенте и сборка подсказки промпта::

    registry = ToolRegistry()
    register_fs_tools(registry)          # корень — текущая папка
    hint = registry.build_hint(intro=FS_TOOLS_INTRO, footer=SANDBOX_NOTE)

Форматирование результата в ``format_display``::

    custom = format_tool_payload(message.content)
    if custom:
        return f"tool[{message.tool_id}]> {custom}"
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional, Union

from ..ai import Tool, ToolRegistry  # type: ignore[no-redef]

from .base import FsTool, FsToolError, human_size
from .create_file import CreateFileTool
from .delete_file import DeleteFileTool
from .edit_file import EditFileTool
from .find_file import FindFileTool
from .list_dir import ListDirTool
from .regex_search import RegexSearchTool
from .rename_file import RenameFileTool

logger = logging.getLogger(__name__)

__all__ = [
    "FsTool",
    "FsToolError",
    "human_size",
    "CreateFileTool",
    "DeleteFileTool",
    "EditFileTool",
    "FindFileTool",
    "ListDirTool",
    "RegexSearchTool",
    "RenameFileTool",
    "FS_TOOL_CLASSES",
    "build_fs_tools",
    "register_fs_tools",
]

#: Классы файловых инструментов в порядке регистрации.
FS_TOOL_CLASSES: tuple[type[FsTool], ...] = (
    ListDirTool,
    FindFileTool,
    CreateFileTool,
    DeleteFileTool,
    RenameFileTool,
    EditFileTool,
    RegexSearchTool,
)


def build_fs_tools(root: Optional[Union[str, Path]] = None) -> list[Tool]:
    """Создать экземпляры файловых инструментов.

    Каждый инструмент — наследник :class:`~ai.Tool` (через
    :class:`~base.FsTool`), поэтому экземпляры готовы к регистрации
    в :class:`~ai.ToolRegistry` как есть, без обёртки ``Tool(...)``.

    Args:
        root: Корень песочницы (рабочая директория); ``None`` — текущая
            папка процесса.

    Returns:
        Список экземпляров инструментов (объектов :class:`~ai.Tool`),
        готовых к регистрации.
    """
    return [cls(root=root) for cls in FS_TOOL_CLASSES]


def register_fs_tools(
    registry: ToolRegistry,
    root: Optional[Union[str, Path]] = None,
) -> list[str]:
    """Зарегистрировать файловые инструменты в реестре агента.

    Args:
        registry: Реестр инструментов агента.
        root: Корень песочницы; ``None`` — текущая папка процесса.

    Returns:
        Список имён зарегистрированных инструментов.
    """
    names: list[str] = []
    for instance in build_fs_tools(root=root):
        registry.register(instance)
        names.append(instance.name)
    return names
