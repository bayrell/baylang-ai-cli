"""Обёртка для запуска пакета: ``python -m src``."""

import sys

from .main import main

if __name__ == "__main__":
    sys.exit(main())
