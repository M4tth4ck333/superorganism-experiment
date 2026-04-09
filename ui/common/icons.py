from __future__ import annotations

from PySide6.QtCore import QSize
from PySide6.QtGui import QIcon

import resources_rc


def icon(name: str) -> QIcon:
    return QIcon(f":/icons/{name}.svg")


def icon_size(px: int) -> QSize:
    return QSize(px, px)
