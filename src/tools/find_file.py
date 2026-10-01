"""Инструмент ``fs_find`` — поиск файлов и папок по шаблону имени.

Класс :class:`FindFileTool` ищет записи рабочей директории по glob-шаблону
(например ``*.py`` или ``report_*.txt``) с рекурсивным обходом. Служебные
папки и закрытые файлы (``.env``) не рассматриваются.
"""

from __future__ import annotations

import fnmatch
from typing import Any

from .base import MAX_LIST_ENTRIES, FsTool, FsToolError

__all__ = ["FindFileTool"]


class FindFileTool(FsTool):
    """Инструмент поиска файлов по имени (``fs_find``).

    Шаблон сопоставляется с именем записи (не с полным путём):
    ``*`` заменяет любую часть имени, ``?`` — один символ. Поиск
    рекурсивный от указанной папки.
    """

    NAME = "fs_find"
    DESCRIPTION = (
        "Найти файлы и папки в рабочей директории проекта по шаблону имени "
        "(glob). Примеры шаблонов: '*.py', 'report_*.txt', 'test_?'. "
        "Поиск рекурсивный; служебные папки (.git, __pycache__) пропускаются."
    )
    HINT = (
        "поиск файлов и папок по glob-шаблону имени (например '*.py'); "
        "рекурсивно"
    )
    PARAMETERS = {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Шаблон имени файла/папки, например '*.py' или 'config.json'.",
            },
            "path": {
                "type": "string",
                "description": "Папка для поиска относительно рабочей директории. По умолчанию '.'.",
            },
            "show_hidden": {
                "type": "boolean",
                "description": "Искать и среди скрытых записей. По умолчанию false.",
            },
            "max_results": {
                "type": "integer",
                "description": "Максимум результатов (1..500). По умолчанию 50.",
            },
        },
        "required": ["pattern"],
    }

    async def run(  # type: ignore[override]
        self,
        pattern: str,
        path: str = ".",
        show_hidden: bool = False,
        max_results: int = 50,
    ) -> dict[str, Any]:
        """Найти записи, имена которых совпадают с glob-шаблоном.

        Args:
            pattern: Glob-шаблон имени.
            path: Стартовая папка относительно корня песочницы.
            show_hidden: Искать и среди скрытых записей.
            max_results: Максимум путей в ответе.

        Returns:
            Результат ``ok`` с полями ``pattern``, ``matches``, ``total``,
            ``truncated``.

        Raises:
            FsToolError: Если шаблон пуст или путь вне песочницы.
        """
        if not isinstance(pattern, str) or not pattern.strip():
            raise FsToolError("Шаблон поиска (pattern) не может быть пустым")
        pattern = pattern.strip()
        target = self.resolve_path(path, allow_root=True, must_exist=True)
        try:
            limit = max(1, min(int(max_results), MAX_LIST_ENTRIES))
        except (TypeError, ValueError):
            limit = 50

        matches: list[dict[str, Any]] = []
        total = 0
        truncated = False
        for item, is_dir in self.iter_tree(target, show_hidden=bool(show_hidden)):
            if not fnmatch.fnmatch(item.name, pattern):
                continue
            total += 1
            if len(matches) >= limit:
                truncated = True
                continue
            matches.append(
                {"path": self.rel(item), "type": "dir" if is_dir else "file"}
            )
        matches.sort(key=lambda e: (e["type"] != "dir", e["path"]))
        return self.ok(
            pattern=pattern,
            path=self.rel(target),
            matches=matches,
            total=total,
            shown=len(matches),
            truncated=truncated,
        )

    @staticmethod
    def format_message(data: dict[str, Any]) -> str:
        """Отформатировать результаты поиска для ``format_display``.

        Args:
            data: Результат :meth:`run` (``ok`` или ``error``).

        Returns:
            Многострочная строка со списком найденных путей.
        """
        if data.get("status") != "ok":
            return f"❌ fs_find: {data.get('error', 'неизвестная ошибка')}"
        lines = [
            f"🔍 fs_find '{data.get('pattern', '?')}' в {data.get('path', '.')} — "
            f"найдено: {data.get('total', 0)}"
        ]
        for entry in data.get("matches", []):
            marker = "📁" if entry.get("type") == "dir" else "📄"
            suffix = "/" if entry.get("type") == "dir" else ""
            lines.append(f"  {marker} {entry.get('path', '?')}{suffix}")
        if data.get("truncated"):
            lines.append("  … список усечён — увеличьте max_results")
        if not data.get("matches"):
            lines.append("  (ничего не найдено — попробуйте другой шаблон, например '*')")
        return "\n".join(lines)
