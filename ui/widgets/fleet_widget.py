from __future__ import annotations

import datetime
from typing import Any

from PyQt6.QtCore import Qt, QAbstractTableModel, QModelIndex, QSortFilterProxyModel
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QTableView, QFrame, QHeaderView,
)

COLUMNS = ["Name", "IP", "Commit", "Uptime", "Disk", "BTC", "BTC Address", "Runway", "Region", "Last Seen"]


def _fmt_uptime(seconds: int) -> str:
    h = seconds // 3600
    m = (seconds % 3600) // 60
    return f"{h}h {m}m"


def _fmt_disk(used_bytes: int, total_bytes: int) -> str:
    used_gb  = used_bytes  / (1024 ** 3)
    total_gb = total_bytes / (1024 ** 3)
    return f"{used_gb:.1f}/{total_gb:.1f} GB"


def _fmt_last_seen(ts: int) -> str:
    return datetime.datetime.fromtimestamp(ts).strftime("%H:%M:%S")


class FleetTableModel(QAbstractTableModel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._rows: list[dict] = []

    def load(self, rows: list[dict]) -> None:
        self.beginResetModel()
        self._rows = rows
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()) -> int:
        return len(self._rows)

    def columnCount(self, parent=QModelIndex()) -> int:
        return len(COLUMNS)

    def headerData(self, section: int, orientation: Qt.Orientation, role=Qt.ItemDataRole.DisplayRole) -> Any:
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            return COLUMNS[section]
        return None

    def data(self, index: QModelIndex, role=Qt.ItemDataRole.DisplayRole) -> Any:
        if not index.isValid():
            return None
        row = self._rows[index.row()]
        col = index.column()

        if role == Qt.ItemDataRole.DisplayRole:
            if col == 0:
                return row.get("friendly_name", "")
            elif col == 1:
                return row.get("public_ip", "")
            elif col == 2:
                commit = row.get("git_commit_hash", "")
                return commit[:8] if commit else ""
            elif col == 3:
                return _fmt_uptime(row.get("uptime_seconds", 0))
            elif col == 4:
                return _fmt_disk(row.get("disk_used_bytes", 0), row.get("disk_total_bytes", 0))
            elif col == 5:
                return f"{row.get('btc_balance_sat', 0)} sat"
            elif col == 6:
                addr = row.get("btc_address", "")
                return addr[:10] + "\u2026" if len(addr) > 10 else addr
            elif col == 7:
                return f"{row.get('vps_days_remaining', 0)}d"
            elif col == 8:
                return row.get("vps_provider_region", "")
            elif col == 9:
                ts = row.get("last_seen", 0)
                return _fmt_last_seen(ts) if ts else ""

        if role == Qt.ItemDataRole.TextAlignmentRole:
            return int(Qt.AlignmentFlag.AlignCenter)

        if role == Qt.ItemDataRole.ForegroundRole:
            if col == 7:  # Runway
                days = row.get("vps_days_remaining", 0)
                if days > 30:
                    return QColor("#34d399")   # green
                elif days >= 7:
                    return QColor("#fbbf24")   # amber
                else:
                    return QColor("#ef4444")   # red

        return None


class FleetWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(34, 28, 34, 28)
        layout.setSpacing(22)

        header = QHBoxLayout()
        title = QLabel("Fleet")
        title.setObjectName("pageTitle")
        self._node_count_lbl = QLabel("Nodes: 0")
        self._node_count_lbl.setObjectName("nodeCountBadge")
        header.addWidget(title)
        header.addStretch()
        header.addWidget(self._node_count_lbl)

        table_card = QFrame()
        table_card.setObjectName("tableCard")
        card_layout = QVBoxLayout(table_card)
        card_layout.setContentsMargins(0, 0, 0, 0)

        self._model = FleetTableModel()
        self._proxy = QSortFilterProxyModel()
        self._proxy.setSourceModel(self._model)

        self._table = QTableView()
        self._table.setModel(self._proxy)
        self._table.setSortingEnabled(True)
        self._table.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        self._table.setShowGrid(False)
        self._table.setAlternatingRowColors(True)
        self._table.setMouseTracking(True)
        self._table.verticalHeader().hide()
        self._table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        hdr = self._table.horizontalHeader()
        hdr.setStretchLastSection(False)
        hdr.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)      # cols 1-8 divide remaining space equally
        hdr.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)     # Name — fixed, same as Infohash
        hdr.setSectionResizeMode(9, QHeaderView.ResizeMode.Fixed)     # Last Seen — fixed, same as Last Check
        self._table.setColumnWidth(0, 115)
        self._table.setColumnWidth(9, 162)

        card_layout.addWidget(self._table)
        self._table.clicked.connect(self._on_cell_clicked)
        self._proxy.sort(7, Qt.SortOrder.DescendingOrder)

        stats_bar = QHBoxLayout()
        stats_bar.setSpacing(24)
        self._total_lbl    = QLabel("Total: 0")
        self._safe_lbl     = QLabel("Safe: 0")
        self._safeish_lbl  = QLabel("Safe-ish: 0")
        self._dying_lbl    = QLabel("Dying: 0")
        self._total_lbl.setObjectName("statTotal")
        self._safe_lbl.setObjectName("statSafe")
        self._safeish_lbl.setObjectName("statSafeish")
        self._dying_lbl.setObjectName("statDying")
        for lbl in (self._total_lbl, self._safe_lbl, self._safeish_lbl, self._dying_lbl):
            stats_bar.addWidget(lbl)
        stats_bar.addStretch()

        layout.addLayout(header)
        layout.addLayout(stats_bar)
        layout.addWidget(table_card, 1)

        self._table.setStyleSheet("""
QTableView {
    background: transparent;
    color: #e5e7eb;
    border: none;
    gridline-color: transparent;
    alternate-background-color: rgba(59,130,246,0.04);
    selection-background-color: rgba(59,130,246,0.20);
    selection-color: white;
}
QTableView::item:hover {
    background: rgba(59,130,246,0.08);
}
QHeaderView::section {
    background: rgba(148,163,184,0.35);
    color: white;
    border: none;
    padding: 16px 14px;
    font-weight: 600;
}
QScrollBar:vertical { background: transparent; width: 6px; margin: 0; }
QScrollBar::handle:vertical { background: rgba(148,163,184,0.25); border-radius: 3px; min-height: 24px; }
QScrollBar::handle:vertical:hover { background: rgba(148,163,184,0.45); }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar:horizontal { background: transparent; height: 6px; margin: 0; }
QScrollBar::handle:horizontal { background: rgba(148,163,184,0.25); border-radius: 3px; min-width: 24px; }
QScrollBar::handle:horizontal:hover { background: rgba(148,163,184,0.45); }
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }
""")
        self.setStyleSheet("""
#nodeCountBadge {
    background: rgba(59,130,246,0.12);
    color: #93c5fd;
    border-radius: 10px;
    padding: 4px 12px;
    font-size: 13px;
    font-weight: 600;
}
#statTotal    { background: rgba(255,255,255,0.06); color: #cbd5e1; border-radius: 10px; padding: 4px 12px; font-size: 13px; font-weight: 600; }
#statSafe     { background: rgba(52,211,153,0.12);  color: #34d399; border-radius: 10px; padding: 4px 12px; font-size: 13px; font-weight: 600; }
#statSafeish  { background: rgba(251,146,60,0.12);  color: #fb923c; border-radius: 10px; padding: 4px 12px; font-size: 13px; font-weight: 600; }
#statDying    { background: rgba(239,68,68,0.12);   color: #ef4444; border-radius: 10px; padding: 4px 12px; font-size: 13px; font-weight: 600; }
""")

    def _on_cell_clicked(self, index) -> None:
        source = self._proxy.mapToSource(index)
        row = self._model._rows[source.row()]
        col = source.column()
        if col == 6:  # BTC Address
            QApplication.clipboard().setText(row.get("btc_address", ""))

    def load(self, fleet: dict) -> None:
        rows = sorted(fleet.values(), key=lambda x: x.get("friendly_name", ""))
        self._model.load(rows)
        self._node_count_lbl.setText(f"Nodes: {len(rows)}")

        total   = len(rows)
        safe    = sum(1 for r in rows if r.get("vps_days_remaining", 0) > 60)
        safeish = sum(1 for r in rows if 30 <= r.get("vps_days_remaining", 0) <= 60)
        dying   = sum(1 for r in rows if r.get("vps_days_remaining", 0) < 30)

        self._total_lbl.setText(f"Total: {total}")
        self._safe_lbl.setText(f"Safe: {safe}")
        self._safeish_lbl.setText(f"Safe-ish: {safeish}")
        self._dying_lbl.setText(f"Dying: {dying}")
