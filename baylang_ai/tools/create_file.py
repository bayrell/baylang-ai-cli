"""Инструмент ``fs_create`` — создание файла внутри рабочей директории.

Класс :class:`CreateFileTool` создаёт (или перезаписывает по запросу)
файл с заданным содержимым; недостающие папки создаются автоматически.
Выход за пределы песочницы и запись в закрытые файлы (``.env``) запрещены.
"""

from __future__ import annotations

from typing import Any

from .base import MAX_TEXT_SIZE, FsTool, FsToolError, human_size

__all__ = ["CreateFileTool"]


class CreateFileTool(FsTool):
    """Инструмент создания файла (``fs_create``).

    По умолчанию отказался бы затирать существующий файл: для перезаписи
    нужно явно передать ``overwrite=true``. Содержимое ограничено
    :data:`~base.MAX_TEXT_SIZE` (1 МБ).
    """

    NAME = "fs_create"
    DESCRIPTION = (
        "Создать новый файл в рабочей директории проекта с заданным содержимым. "
        "Недостающие папки создаются автоматически. Существующий файл не "
        "перезаписывается без overwrite=true."
    )
    HINT = (
        "создать новый файл с содержимым (недостающие папки создаются; "
        "существующий не перезаписывается без overwrite=true)"
    )
    PARAMETERS = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Путь к файлу относительно рабочей директории, например 'notes/todo.md'.",
            },
            "content": {
                "type": "string",
                "description": "Содержимое файла (UTF-8). По умолчанию пустое.",
            },
            "overwrite": {
                "type": "boolean",
                "description": "Перезаписать файл, если он уже существует. По умолчанию false.",
            },
            "create_dirs": {
                "type": "boolean",
                "description": "Создать недостающие папки пути. По умолчанию true.",
            },
        },
        "required": ["path"],
    }

    async def run(  # type: ignore[override]
        self,
        path: str,
        content: str = "",
        overwrite: bool = False,
        create_dirs: bool = True,
    ) -> dict[str, Any]:
        """Создать файл с указанным содержимым.

        Args:
            path: Путь к файлу относительно корня песочницы.
            content: Содержимое (UTF-8).
            overwrite: Разрешить перезапись существующего файла.
            create_dirs: Создавать недостающие папки.

        Returns:
            Результат ``ok`` с полями ``path``, ``bytes``, ``created`` /
            ``overwritten``.

        Raises:
            FsToolError: При выходе за песочницу, слишком большом
                содержимом, записи в папку или отказе перезаписи.
        """
        if content is None:
            content = ""
        if not isinstance(content, str):
            raise FsToolError("Содержимое (content) должно быть строкой")
        encoded = content.encode("utf-8")
        if len(encoded) > MAX_TEXT_SIZE:
            raise FsToolError(
                f"Содержимое слишком большое ({human_size(len(encoded))}, "
                f"лимит {human_size(MAX_TEXT_SIZE)})"
            )
        target = self.resolve_path(path, allow_root=False)
        if target.is_dir():
            raise FsToolError(
                f"{self.rel(target)!r} — папка; укажите путь к файлу внутри неё"
            )
        existed = target.exists()
        if existed and not overwrite:
            raise FsToolError(
                f"{self.rel(target)!r} уже существует — передайте overwrite=true для перезаписи"
            )
        parent = target.parent
        if not parent.exists():
            if not create_dirs:
                raise FsToolError(
                    f"Папка {self.rel(parent)!r} не существует — передайте create_dirs=true"
                )
            parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return "File overwritten" if existed else "File created"

    @staticmethod
    def format_message(params: dict[str, Any], data: Any) -> str:
        """Отформатировать результат создания файла для ``format_display``.

        Args:
            data: Результат :meth:`run` (``ok`` или ``error``).

        Returns:
            Однострочное сообщение о созданном/перезаписанном файле.
        """
        if data == "File overwritten":
            return f"перезаписан {params.get('path', '?')}"
        return f"создан {params.get('path', '?')}"
