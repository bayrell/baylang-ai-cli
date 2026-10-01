"""Инструмент ``fs_search_regex`` — поиск по регулярному выражению.

Класс :class:`RegexSearchTool` ищет совпадения regexp в содержимом файлов
рабочей директории (аналог grep): возвращает путь, номер строки и текст.
Бинарные и слишком большие файлы пропускаются, служебные папки не
просматриваются, поиск ограничен песочницей.
"""

from __future__ import annotations

import fnmatch
import re
from typing import Any

from .base import MAX_FILE_SIZE, MAX_LIST_ENTRIES, FsTool, FsToolError

__all__ = ["RegexSearchTool"]


class RegexSearchTool(FsTool):
    """Инструмент поиска по regexp (``fs_search_regex``).

    Регулярное выражение применяется построчно (синтаксис Python ``re``);
    в результат попадают номера строк — удобно для последующего fs_edit.
    """

    NAME = "fs_search_regex"
    DESCRIPTION = (
        "Поиск по регулярному выражению (regexp, синтаксис Python re) по "
        "содержимому файлов рабочей директории проекта — как grep: путь, "
        "номер строки, текст. Можно ограничить папку и шаблон имён файлов "
        "(glob). Бинарные и файлы больше 1 МБ пропускаются."
    )
    PARAMETERS = {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Регулярное выражение для поиска, например 'def test_\\w+' или 'TODO'.",
            },
            "path": {
                "type": "string",
                "description": "Папка для поиска относительно рабочей директории. По умолчанию '.'.",
            },
            "glob": {
                "type": "string",
                "description": "Шаблон имён файлов, например '*.py'. По умолчанию '*' (все файлы).",
            },
            "case_insensitive": {
                "type": "boolean",
                "description": "Игнорировать регистр. По умолчанию false.",
            },
            "show_hidden": {
                "type": "boolean",
                "description": "Искать и в скрытых файлах. По умолчанию false.",
            },
            "max_results": {
                "type": "integer",
                "description": "Максимум совпадений в ответе (1..500). По умолчанию 100.",
            },
        },
        "required": ["pattern"],
    }

    async def run(  # type: ignore[override]
        self,
        pattern: str,
        path: str = ".",
        glob: str = "*",
        case_insensitive: bool = False,
        show_hidden: bool = False,
        max_results: int = 100,
    ) -> dict[str, Any]:
        """Найти строки файлов, содержащие совпадение регулярного выражения.

        Args:
            pattern: Регулярное выражение (Python ``re``).
            path: Папка для поиска относительно корня песочницы.
            glob: Шаблон имён файлов для фильтра.
            case_insensitive: Поиск без учёта регистра.
            show_hidden: Просматривать и скрытые файлы.
            max_results: Максимум совпадений в ответе.

        Returns:
            Результат ``ok`` с полями ``matches`` (``path``/``line``/
            ``text``), ``matches_total``, ``files_searched``,
            ``files_skipped``, ``truncated``.

        Raises:
            FsToolError: При некорректном regexp или пути вне песочницы.
        """
        if not isinstance(pattern, str) or not pattern.strip():
            raise FsToolError("Регулярное выражение (pattern) не может быть пустым")
        flags = re.IGNORECASE if case_insensitive else 0
        try:
            regex = re.compile(pattern, flags)
        except re.error as exc:
            raise FsToolError(f"Некорректное регулярное выражение {pattern!r}: {exc}") from exc
        target = self.resolve_path(path, allow_root=True, must_exist=True)
        try:
            limit = max(1, min(int(max_results), MAX_LIST_ENTRIES))
        except (TypeError, ValueError):
            limit = 100

        matches: list[dict[str, Any]] = []
        matches_total = 0
        files_searched = 0
        files_skipped = 0
        truncated = False
        name_filter = glob.strip() if isinstance(glob, str) and glob.strip() else "*"

        for item, is_dir in self.iter_tree(target, show_hidden=bool(show_hidden)):
            if is_dir:
                continue
            if not fnmatch.fnmatch(item.name, name_filter):
                continue
            try:
                raw = item.read_bytes()
            except OSError:
                files_skipped += 1
                continue
            if len(raw) > MAX_FILE_SIZE or b"\x00" in raw[:4096]:
                files_skipped += 1
                continue
            text = raw.decode("utf-8", errors="replace")
            files_searched += 1
            for number, line in enumerate(text.splitlines(), start=1):
                for found in regex.finditer(line):
                    matches_total += 1
                    if len(matches) >= limit:
                        truncated = True
                        continue
                    matches.append(
                        {
                            "path": self.rel(item),
                            "line": number,
                            "text": line.strip()[:300],
                            "match": found.group(0)[:100],
                        }
                    )
        return self.ok(
            pattern=pattern,
            path=self.rel(target),
            glob=name_filter,
            matches=matches,
            matches_total=matches_total,
            files_searched=files_searched,
            files_skipped=files_skipped,
            truncated=truncated,
        )

    @staticmethod
    def format_message(data: dict[str, Any]) -> str:
        """Отформатировать результаты поиска по regexp для ``format_display``.

        Args:
            data: Результат :meth:`run` (``ok`` или ``error``).

        Returns:
            Многострочная строка с совпадениями вида ``path:line: текст``.
        """
        if data.get("status") != "ok":
            return f"❌ fs_search_regex: {data.get('error', 'неизвестная ошибка')}"
        lines = [
            f"🔍 regexp '{data.get('pattern', '?')}' в {data.get('path', '.')} "
            f"[{data.get('glob', '*')}] — совпадений: {data.get('matches_total', 0)} "
            f"(файлов просмотрено: {data.get('files_searched', 0)})"
        ]
        for entry in data.get("matches", []):
            text = entry.get("text", "")
            lines.append(f"  {entry.get('path', '?')}:{entry.get('line', 0)}: {text}")
        if data.get("truncated"):
            lines.append("  … результаты усечены — увеличьте max_results или уточните glob/path")
        if not data.get("matches"):
            lines.append("  (совпадений нет — проверьте pattern или расширьте glob)")
        skipped = int(data.get("files_skipped", 0))
        if skipped:
            lines.append(f"  (пропущено файлов: {skipped} — бинарные или больше 1 МБ)")
        return "\n".join(lines)
