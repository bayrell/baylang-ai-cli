"""Инструмент ``fs_read`` — чтение текстового файла рабочей директории.

Класс :class:`ReadFileTool` читает файл внутри песочницы (UTF-8) и
возвращает его содержимое модели. Поддерживается чтение диапазона строк
(``offset`` / ``limit``) — так большие файлы можно изучать частями,
не переполняя контекст. Лимит размера — :data:`~base.MAX_FILE_SIZE` (1 МБ),
закрытые файлы (``.env``) и выход за рабочую директорию запрещены.
"""

from __future__ import annotations

from typing import Any

from .base import MAX_LIST_ENTRIES, FsTool, human_size

__all__ = ["ReadFileTool"]


class ReadFileTool(FsTool):
    """Инструмент чтения файла (``fs_read``).
    """

    NAME = "fs_read"
    DESCRIPTION = (
        "Прочитать текстовый файл (UTF-8) в рабочей директории проекта и "
        "получить его содержимое."
    )
    HINT = (
        "прочитать текстовый файл целиком"
    )
    PARAMETERS = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Путь к файлу относительно рабочей директории, например 'README.md'.",
            },
        },
        "required": ["path"],
    }

    async def run(  # type: ignore[override]
        self,
        path: str
    ) -> dict[str, Any]:
        """Прочитать файл (или диапазон строк) внутри песочницы.

        Args:
            path: Путь к файлу относительно корня песочницы.
            offset: Номер первой строки (с 1); значения меньше 1
                приравниваются к 1.
            limit: Максимум строк в ответе (ограничен
                :data:`~base.MAX_LIST_ENTRIES`).

        Returns:
            Результат ``ok`` с полями ``path``, ``content``,
            ``offset``, ``lines``, ``total_lines``, ``size`` и
            ``truncated``.

        Raises:
            FsToolError: Если путь вне песочницы, не существует,
                является папкой, слишком большой или не является
                текстовым UTF-8.
        """
        target = self.resolve_path(path,
            allow_root=False, must_exist=True
        )
        text = self.read_text(target)
        return text

    @staticmethod
    def format_message(params: dict[str, Any], data: Any) -> str:
        """Отформатировать результат чтения файла для ``format_display``.

        Args:
            data: Результат :meth:`run` (``ok`` или ``error``).

        Returns:
            Однострочное сообщение о прочитанном файле: путь, число
            строк и размер; для ошибки — текст ошибки.
        """
        return f"Чтение {params.get('path')}"
