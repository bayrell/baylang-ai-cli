#!/usr/bin/env python3
"""Скрипт установки пакета BayLang AI (setuptools).

Каталог ``src/`` устанавливается как Python-пакет ``baylang_ai``
(см. ``package_dir``): ядро ``baylang_ai.ai``, точка входа
``baylang_ai.main`` и файловые инструменты ``baylang_ai.tools``.
Внутренние импорты модулей работают в обоих режимах: установленный пакет
(относительные импорты) и прямой запуск из репозитория
(``python src/main.py``, ``bin/baylang_ai`` — top-level импорты).

Команды установки::

    pip install .          # установить пакет и консольные команды
    pip install -e .       # режим разработки (editable install)
    pip install .[dev]     # плюс тестовые зависимости (pytest)

После установки доступны:

* ``baylang_ai`` — скрипт запуска ``bin/baylang_ai`` (через ``scripts``);
* ``baylang-ai`` — console_scripts точка входа ``baylang_ai.main:main``;
* Python-пакет ``baylang_ai`` для импорта в своем коде.

Сборка дистрибутива (нужен ``pip install build``)::

    python -m build         # wheel и sdist в каталог dist/
"""

from __future__ import annotations

from pathlib import Path

from setuptools import setup

ROOT = Path(__file__).resolve().parent

setup(
    name="baylang-ai",
    version="1.0.0",
    description=(
        "CLI и библиотека AI-агентов поверх OpenRouter "
        "(OpenAI-совместимый chat-completions API)"
    ),
    long_description=(ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    license="Apache-2.0",
    # Пакет baylang_ai живет в каталоге src/ (ядро ai.py, CLI main.py,
    # файловые инструменты tools/).
    packages=["baylang_ai", "baylang_ai.tools"],
    package_dir={"baylang_ai": "src"},
    scripts=["bin/baylang_ai"],
    python_requires=">=3.12",
    install_requires=[
        "httpx>=0.27,<1",
        "python-dotenv>=1.2.0",
    ],
    extras_require={
        "dev": ["pytest>=8,<9"],
    },
    entry_points={
        "console_scripts": [
            "baylang-ai=baylang_ai.main:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.12",
        "Topic :: Communications :: Chat",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    zip_safe=False,
)
