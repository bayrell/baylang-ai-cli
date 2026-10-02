"""Инструмент ``fs_edit`` — редактирование файлов рабочей директории.

Класс :class:`EditFileTool` работает в двух режимах:

* **замена** — если задан ``old_text``: ищет фрагмент и заменяет на
  ``new_text`` (одно вхождение; ``replace_all=true`` — все);
* **запись** — если ``old_text`` пуст: создаёт/перезаписывает файл
  содержимым ``new_text`` целиком.

Запись и чтение ограничены лимитами размера, выход за песочницу и
редактирование закрытых файлов (``.env``) запрещены.
"""

from __future__ import annotations

from typing import Any

from .base import (
    MAX_TEXT_SIZE,
    FsTool,
    FsToolError,
    human_size,
)

__all__ = ["EditFileTool"]


class EditFileTool(FsTool):
    """Инструмент редактирования файла (``fs_edit``).

    Для точечной правки передайте уникальный ``old_text`` (иначе получите
    ошибку с числом вхождений — так правка не промахнётся мимо цели).
    """

    NAME = "fs_edit"
    DESCRIPTION = (
        "Отредактировать файл в рабочей директории проекта. Если задан old_text — "
        "заменяет его на new_text в существующем файле (одно вхождение; "
        "replace_all=true заменяет все). Если old_text пуст — создаёт файл "
        "или целиком перезаписывает его содержимым new_text."
    )
    HINT = (
        "отредактировать файл: заменить фрагмент old_text на new_text "
        "(или перезаписать файл целиком, если old_text пуст)"
    )
    PARAMETERS = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Путь к файлу относительно рабочей директории.",
            },
            "new_text": {
                "type": "string",
                "description": "Новый текст: замена old_text либо полное содержимое файла.",
            },
            "old_text": {
                "type": "string",
                "description": "Фрагмент для замены. Пусто — режим полной записи файла.",
            },
            "replace_all": {
                "type": "boolean",
                "description": "Заменить все вхождения old_text. По умолчанию false (только первое).",
            },
        },
        "required": ["path", "new_text"],
    }

    async def run(  # type: ignore[override]
        self,
        path: str,
        new_text: str = "",
        old_text: str = "",
        replace_all: bool = False,
    ) -> dict[str, Any]:
        """Заменить фрагмент файла или записать файл целиком.

        Args:
            path: Путь к файлу относительно корня песочницы.
            new_text: Текст замены или полное содержимое.
            old_text: Фрагмент для замены; пусто — режим записи.
            replace_all: Заменять все вхождения.

        Returns:
            Результат ``ok``: для замены — ``mode="replace"`` и
            ``replacements``; для записи — ``mode="write"``, ``created``
            и ``bytes``.

        Raises:
            FsToolError: При выходе за песочницу, отсутствии файла,
                ненайденном/неуникальном ``old_text`` и превышении лимитов.
        """
        if new_text is None:
            new_text = ""
        if old_text is None:
            old_text = ""
        if not isinstance(new_text, str) or not isinstance(old_text, str):
            raise FsToolError("old_text и new_text должны быть строками")
        target = self.resolve_path(path, allow_root=False)

        if old_text:
            text = self.read_text(target)
            count = text.count(old_text)
            if count == 0:
                raise FsToolError(
                    "old_text не найден в файле — проверьте фрагмент по fs_search_regex"
                )
            if count > 1 and not replace_all:
                raise FsToolError(
                    f"old_text встречается в файле {count} раз(а) — уточните фрагмент "
                    "или передайте replace_all=true"
                )
            updated = (
                text.replace(old_text, new_text)
                if replace_all
                else text.replace(old_text, new_text, 1)
            )
            target.write_text(updated, encoding="utf-8")
            return self.ok(
                path=self.rel(target),
                mode="replace",
                replacements=(count if replace_all else 1),
                bytes=len(updated.encode("utf-8")),
            )

        encoded = new_text.encode("utf-8")
        if len(encoded) > MAX_TEXT_SIZE:
            raise FsToolError(
                f"Содержимое слишком большое ({human_size(len(encoded))}, "
                f"лимит {human_size(MAX_TEXT_SIZE)})"
            )
        if target.is_dir():
            raise FsToolError(
                f"{self.rel(target)!r} — папка; укажите путь к файлу внутри неё"
            )
        created = not target.exists()
        target.write_text(new_text, encoding="utf-8")
        return self.ok(
            path=self.rel(target),
            mode="write",
            created=created,
            bytes=len(encoded),
        )

    @staticmethod
    def format_message(data: dict[str, Any]) -> str:
        """Отформатировать результат редактирования для ``format_display``.

        Args:
            data: Результат :meth:`run` (``ok`` или ``error``).

        Returns:
            Однострочное сообщение о замене или записи файла.
        """
        if data.get("status") != "ok":
            return f"❌ fs_edit: {data.get('error', 'неизвестная ошибка')}"
        path = data.get("path", "?")
        size = human_size(int(data.get("bytes", 0)))
        if data.get("mode") == "replace":
            return f"✂️ fs_edit: в {path} заменено вхождений — {data.get('replacements', 0)} ({size})"
        if data.get("created"):
            return f"✅ fs_edit: создан {path} ({size})"
        return f"✅ fs_edit: перезаписан {path} ({size})"
