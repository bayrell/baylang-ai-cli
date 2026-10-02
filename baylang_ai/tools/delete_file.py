"""Инструмент ``fs_delete`` — удаление файлов и папок рабочей директории.

Класс :class:`DeleteFileTool` удаляет файл или папку внутри песочницы.
Удаление непустой папки требует явного ``recursive=true``; сама рабочая
директория и закрытые файлы (``.env``) защищены от удаления.
"""

from __future__ import annotations

import shutil
from typing import Any

from .base import FsTool, FsToolError

__all__ = ["DeleteFileTool"]


class DeleteFileTool(FsTool):
    """Инструмент удаления файлов и папок (``fs_delete``).

    Мера предосторожности: непустая папка удаляется только с флагом
    ``recursive=true``; удаление самой рабочей директории запрещено
    на уровне песочницы.
    """

    NAME = "fs_delete"
    DESCRIPTION = (
        "Удалить файл или папку в рабочей директории проекта. Непустая папка "
        "удаляется только с recursive=true. Удаление необратимо — "
        "используйте после подтверждения пользователем."
    )
    HINT = (
        "удалить файл или папку (непустая папка — только с recursive=true; "
        "удаление необратимо)"
    )
    PARAMETERS = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Путь к файлу или папке относительно рабочей директории.",
            },
            "recursive": {
                "type": "boolean",
                "description": "Удалить папку вместе со всем содержимым. По умолчанию false.",
            },
        },
        "required": ["path"],
    }

    async def run(  # type: ignore[override]
        self,
        path: str,
        recursive: bool = False,
    ) -> dict[str, Any]:
        """Удалить файл или папку внутри песочницы.

        Args:
            path: Путь удаляемой записи относительно корня песочницы.
            recursive: Разрешить удаление папки с содержимым.

        Returns:
            Результат ``ok`` с полями ``path`` и ``deleted_type``
            (``file`` или ``dir``).

        Raises:
            FsToolError: При выходе за песочницу, отсутствии пути,
                попытке удалить непустую папку без ``recursive``.
        """
        target = self.resolve_path(path, allow_root=False, must_exist=True)
        if target.is_dir():
            try:
                next(target.iterdir())
                is_empty = False
            except StopIteration:
                is_empty = True
            if not is_empty and not recursive:
                raise FsToolError(
                    f"{self.rel(target)!r} — непустая папка; передайте recursive=true для удаления"
                )
            if is_empty:
                target.rmdir()
            else:
                shutil.rmtree(target)
            return self.ok(path=self.rel(target), deleted_type="dir")
        target.unlink()
        return self.ok(path=self.rel(target), deleted_type="file")

    @staticmethod
    def format_message(data: dict[str, Any]) -> str:
        """Отформатировать результат удаления для ``format_display``.

        Args:
            data: Результат :meth:`run` (``ok`` или ``error``).

        Returns:
            Однострочное сообщение об удалённой записи.
        """
        if data.get("status") != "ok":
            return f"❌ fs_delete: {data.get('error', 'неизвестная ошибка')}"
        path = data.get("path", "?")
        if data.get("deleted_type") == "dir":
            return f"🗑 fs_delete: удалена папка {path}/ вместе с содержимым"
        return f"🗑 fs_delete: удалён файл {path}"
