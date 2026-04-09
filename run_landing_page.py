from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

from ui.landing.landing_page import LandingPageWidget


def main() -> None:
    app = QApplication(sys.argv)

    widget = LandingPageWidget()
    widget.resize(1440, 1100)

    qss_path = Path(__file__).parent / "ui" / "landing" / "landing_page.qss"
    if qss_path.exists():
        with open(qss_path, "r", encoding="utf-8") as f:
            app.setStyleSheet(f.read())

    widget.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
