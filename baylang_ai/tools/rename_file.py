"""Инструмент ``fs_rename`` — переименование и перенос файлов/папок.

Класс :class:`RenameFileTool` переименовывает запись или перемещает её
в другую папку, но только внутри рабочей директории: и исходный путь,
и новый обязаны оставаться в песочнице.
"""

from __future__ import annotations

from typing import Any

from .base import FsTool, FsToolError

__all__ = ["RenameFileTool"]


class RenameFileTool(FsTool):
    """Инструмент переименования/перемещения (``fs_rename``).

    Оба пути (старый и новый) разрешаются внутри песочницы; существующий
    файл-приёмник не затирается без ``overwrite=true``.
    """

    NAME = "fs_rename"
    DESCRIPTION = (
        "Переименовать файл/папку или переместить их в другую папку внутри "
        "рабочей директории проекта. Оба пути — относительно рабочей "
        "директории. Существующий файл не затирается без overwrite=true."
    )
    HINT = (
        "переименовать файл/папку или переместить их в другую папку "
        "(существующий не затирается без overwrite=true)"
    )
    PARAMETERS = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Исходный путь к файлу или папке относительно рабочей директории.",
            },
            "new_path": {
                "type": "string",
                "description": "Новый путь относительно рабочей директории, например 'src/new_name.py'.",
            },
            "overwrite": {
                "type": "boolean",
                "description": "Затереть существующий файл по новому пути. По умолчанию false.",
            },
        },
        "required": ["path", "new_path"],
    }

    async def run(  # type: ignore[override]
        self,
        path: str,
        new_path: str,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        """Переименовать или переместить запись.

        Args:
            path: Исходный путь относительно корня песочницы.
            new_path: Новый путь относительно корня песочницы.
            overwrite: Разрешить затирание существующего файла-приёмника.

        Returns:
            Результат ``ok`` с полями ``path``, ``new_path`` и ``type``.

        Raises:
            FsToolError: При выходе за песочницу, отсутствии источника,
                совпадении путей, занятом назначении или отсутствии
                папки-приёмника.
        """
        source = self.resolve_path(path, allow_root=False, must_exist=True)
        target = self.resolve_path(new_path, allow_root=False)
        if source == target:
            raise FsToolError("Старое и новое имена совпадают — делать нечего")
        entry_type = "dir" if source.is_dir() else "file"
        if target.exists():
            if not overwrite:
                raise FsToolError(
                    f"{self.rel(target)!r} уже существует — передайте overwrite=true для замены"
                )
            if target.is_dir():
                raise FsToolError(
                    f"{self.rel(target)!r} — папка; затирать её файлом или папкой нельзя"
                )
            target.unlink()
        if not target.parent.exists():
            raise FsToolError(
                f"Папка {self.rel(target.parent)!r} не существует — сначала создайте её (fs_create)"
            )
        source.rename(target)
        return self.ok(
            path=self.rel(source),
            new_path=self.rel(target),
            type=entry_type,
        )

    @staticmethod
    def format_message(params: dict[str, Any], data: Any) -> str:
        """Отформатировать результат переименования для ``format_display``.

        Args:
            data: Результат :meth:`run` (``ok`` или ``error``).

        Returns:
            Однострочное сообщение «старое → новое».
        """
        if data.get("status") != "ok":
            return f"Error: {data.get('error', 'неизвестная ошибка')}"
        kind = "папка" if data.get("type") == "dir" else "файл"
        return (
            f"Rename {kind} {data.get('path', '?')} → {data.get('new_path', '?')}"
        )
