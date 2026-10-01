"""Инструмент ``fs_list`` — список файлов и папок рабочей директории.

Класс :class:`ListDirTool` выводит содержимое папки внутри песочницы
(по умолчанию — самой рабочей директории), опционально рекурсивно.
Закрытые файлы (``.env``) и служебные каталоги при обходе пропускаются.
"""

from __future__ import annotations

from typing import Any

from .base import MAX_LIST_ENTRIES, FsTool, FsToolError, human_size

__all__ = ["ListDirTool"]


class ListDirTool(FsTool):
    """Инструмент списка файлов и папок (``fs_list``).

    Показывает записи папки: имя, тип (файл/папка) и размер файла.
    Результат усечён лимитом ``max_entries`` (не больше
    :data:`~base.MAX_LIST_ENTRIES`), чтобы не переполнять контекст модели.
    """

    NAME = "fs_list"
    DESCRIPTION = (
        "Показать список файлов и папок в рабочей директории проекта "
        "(или в её подпапке). Пути всегда относительно рабочей директории. "
        "Не рекурсивно по умолчанию; служебные папки (.git, __pycache__) "
        "при рекурсивном обходе пропускаются."
    )
    PARAMETERS = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Папка относительно рабочей директории. По умолчанию '.' (корень проекта).",
            },
            "recursive": {
                "type": "boolean",
                "description": "Рекурсивно обойти подпапки. По умолчанию false.",
            },
            "show_hidden": {
                "type": "boolean",
                "description": "Показывать скрытые записи (имя начинается с '.'). По умолчанию false.",
            },
            "max_entries": {
                "type": "integer",
                "description": "Максимум записей в ответе (1..500). По умолчанию 200.",
            },
        },
        "required": [],
    }

    async def run(  # type: ignore[override]
        self,
        path: str = ".",
        recursive: bool = False,
        show_hidden: bool = False,
        max_entries: int = 200,
    ) -> dict[str, Any]:
        """Собрать список записей папки.

        Args:
            path: Папка относительно корня песочницы.
            recursive: Спускаться в подпапки.
            show_hidden: Включать скрытые записи.
            max_entries: Максимум записей в ответе.

        Returns:
            Результат ``ok`` с полями ``path``, ``entries``, ``total``,
            ``shown``, ``truncated``.

        Raises:
            FsToolError: Если путь вне песочницы, не существует или
                не является папкой.
        """
        target = self.resolve_path(path, allow_root=True, must_exist=True)
        if not target.is_dir():
            raise FsToolError(
                f"{self.rel(target)!r} — файл; его содержимое смотрите через fs_edit или fs_search_regex"
            )
        try:
            limit = max(1, min(int(max_entries), MAX_LIST_ENTRIES))
        except (TypeError, ValueError):
            limit = 200

        entries: list[dict[str, Any]] = []
        total = 0
        truncated = False
        for item, is_dir in self.iter_tree(
            target, recursive=bool(recursive), show_hidden=bool(show_hidden)
        ):
            total += 1
            if len(entries) >= limit:
                truncated = True
                continue
            entry: dict[str, Any] = {
                "name": self.rel(item),
                "type": "dir" if is_dir else "file",
            }
            if not is_dir:
                try:
                    entry["size"] = item.stat().st_size
                except OSError:
                    entry["size"] = -1
            entries.append(entry)
        entries.sort(key=lambda e: (e["type"] != "dir", e["name"]))
        return self.ok(
            path=self.rel(target),
            recursive=bool(recursive),
            entries=entries,
            total=total,
            shown=len(entries),
            truncated=truncated,
        )

    @staticmethod
    def format_message(data: dict[str, Any]) -> str:
        """Отформатировать список папки для ``format_display``.

        Args:
            data: Результат :meth:`run` (``ok`` или ``error``).

        Returns:
            Многострочная строка со списком записей.
        """
        if data.get("status") != "ok":
            return f"❌ fs_list: {data.get('error', 'неизвестная ошибка')}"
        lines = [
            f"📁 {data.get('path', '.')} — записей: {data.get('shown', 0)} из {data.get('total', 0)}"
        ]
        for entry in data.get("entries", []):
            name = entry.get("name", "?")
            if entry.get("type") == "dir":
                lines.append(f"  📁 {name}/")
            else:
                lines.append(f"  📄 {name} ({human_size(int(entry.get('size', -1)))})")
        if data.get("truncated"):
            lines.append("  … список усечён — увеличьте max_entries или уточните path")
        return "\n".join(lines)
