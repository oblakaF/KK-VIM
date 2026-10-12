"""Qt stylesheet and palette generated from theme tokens.

Widgets select styles through object names and dynamic properties (card, role,
variant, status, ...), never through inline colours. Font sizes are in points so
that they scale correctly on HiDPI screens.
"""

from pathlib import Path

from PySide6.QtGui import QColor, QPalette

from kkvim_gui.theme.tokens import SIZES

ICONS = Path(__file__).resolve().parent / "icons"


def build_palette(t):
    """Fusion palette matching the tokens (menus, tooltips, popups not covered by QSS)."""
    p = QPalette()
    roles = {
        QPalette.ColorRole.Window: t["window.bg"],
        QPalette.ColorRole.WindowText: t["text.primary"],
        QPalette.ColorRole.Base: t["surface.bg"],
        QPalette.ColorRole.AlternateBase: t["surface.alt"],
        QPalette.ColorRole.Text: t["text.primary"],
        QPalette.ColorRole.Button: t["surface.bg"],
        QPalette.ColorRole.ButtonText: t["text.primary"],
        QPalette.ColorRole.ToolTipBase: t["surface.bg"],
        QPalette.ColorRole.ToolTipText: t["text.primary"],
        QPalette.ColorRole.PlaceholderText: t["text.disabled"],
        QPalette.ColorRole.Highlight: t["accent"],
        QPalette.ColorRole.HighlightedText: t["accent.fg"],
        QPalette.ColorRole.Link: t["accent"],
    }
    for role, colour in roles.items():
        p.setColor(role, QColor(colour))
    for role in (QPalette.ColorRole.Text, QPalette.ColorRole.WindowText, QPalette.ColorRole.ButtonText):
        p.setColor(QPalette.ColorGroup.Disabled, role, QColor(t["text.disabled"]))
    return p


def build_stylesheet(t):
    s = SIZES
    h = s["control.height"]
    rc = s["radius.control"]
    rk = s["radius.card"]
    chevron = (ICONS / t["icon.chevron"]).as_posix()
    return f"""
QMainWindow, QDialog, QWidget#Page {{
    background: {t["window.bg"]};
}}
QWidget {{
    color: {t["text.primary"]};
}}
QToolTip {{
    background: {t["surface.bg"]}; color: {t["text.primary"]};
    border: 1px solid {t["border.strong"]}; padding: 4px 6px;
}}

/* Application header (GUI_DESIGN.md, 4.1) */
QFrame#AppHeader {{
    background: {t["header.bg"]};
    min-height: {s["header.height"]}px; max-height: {s["header.height"]}px;
}}
QFrame#AppHeader QLabel {{ color: {t["header.fg"]}; background: transparent; }}
QFrame#AppHeader QLabel#HeaderTitle {{ font-size: 11pt; font-weight: 600; }}
QFrame#AppHeader QLabel#HeaderProject {{ color: {t["header.fg.muted"]}; }}
QFrame#AppHeader QFrame#HeaderSeparator {{ background: {t["header.fg.muted"]}; }}
QFrame#AppHeader QToolButton {{
    color: {t["header.fg"]}; background: transparent; border: none;
    border-radius: {rc}px; padding: 4px 10px; font-size: 9.5pt;
}}
QFrame#AppHeader QToolButton:hover {{ background: {t["header.hover"]}; }}
QFrame#AppHeader QToolButton:disabled {{ color: {t["header.fg.muted"]}; }}
QFrame#AppHeader QToolButton[nav="true"] {{
    border-radius: 0; padding: 4px 12px; border-bottom: 2px solid transparent;
    color: {t["header.fg.muted"]};
}}
QFrame#AppHeader QToolButton[nav="true"]:checked {{
    color: {t["header.fg"]}; font-weight: 600; border-bottom: 2px solid {t["header.fg"]};
}}
QFrame#AppHeader QToolButton[nav="true"]:disabled {{ color: {t["header.fg.muted"]}; }}

/* Help: tooltip popup, "i" buttons, help centre (GUI_DESIGN.md, 17) */
QFrame#HelpPopup {{
    background: {t["surface.bg"]}; border: 1px solid {t["border.strong"]}; border-radius: {rc}px;
}}
QFrame#HelpPopup QLabel {{ background: transparent; color: {t["text.primary"]}; }}
QLabel#HelpPopupTitle {{ font-weight: 600; }}
QLabel#HelpPopupHint {{ color: {t["text.secondary"]}; font-size: 8pt; }}
QToolButton#InfoButton {{
    background: transparent; border: none; color: {t["accent"]};
    padding: 0 2px; font-size: 9.5pt;
}}
QToolButton#InfoButton:hover {{ color: {t["accent.hover"]}; }}
QFrame#HelpIndex {{ background: {t["surface.bg"]}; border-right: 1px solid {t["border"]}; }}
QTreeWidget {{ background: {t["surface.bg"]}; border: none; }}
QTreeWidget::item {{ padding: 3px 2px; }}
QTreeWidget::item:selected {{ background: {t["accent.soft"]}; color: {t["text.primary"]}; }}
QTextBrowser#HelpArticle {{ background: {t["surface.bg"]}; border: none; }}

QLabel#TestDataBadge {{
    background: {t["testdata.bg"]}; color: {t["testdata.fg"]};
    border: 1px solid {t["testdata.border"]}; border-radius: {rc}px;
    padding: 2px 8px; font-size: 8.5pt; font-weight: 600;
}}

/* Cards */
QFrame[card="true"] {{
    background: {t["surface.bg"]};
    border: 1px solid {t["border"]}; border-radius: {rk}px;
}}
QFrame[card="true"] QLabel {{ background: transparent; border: none; }}
QLabel[role="cardTitle"] {{ font-size: 10pt; font-weight: 600; }}
QLabel[role="pageTitle"] {{ font-size: 13pt; font-weight: 600; }}
QLabel[role="secondary"] {{ color: {t["text.secondary"]}; font-size: 9pt; }}
QLabel[role="fieldLabel"] {{ color: {t["text.secondary"]}; font-size: 9pt; }}
QLabel[role="metricLabel"] {{ color: {t["text.secondary"]}; font-size: 8.5pt; }}
QLabel[role="metricValue"] {{ font-size: 13pt; font-weight: 600; }}
QLabel[role="tag"] {{
    color: {t["text.secondary"]}; background: {t["surface.alt"]};
    border: 1px solid {t["border"]}; border-radius: {rc}px;
    padding: 1px 6px; font-size: 8pt;
}}
QFrame[note="info"] {{
    background: {t["surface.alt"]}; border: none; border-radius: {rc}px;
}}
QFrame[note="testdata"] {{
    background: {t["testdata.bg"]}; border: 1px solid {t["testdata.border"]}; border-radius: {rc}px;
}}
QFrame[note="testdata"] QLabel {{ color: {t["testdata.fg"]}; }}
QFrame[note="lock"] {{
    background: {t["accent.soft"]}; border: 1px solid {t["accent"]};
    border-left: 4px solid {t["accent"]}; border-radius: {rc}px;
}}
QFrame[note="lock"] QLabel {{ color: {t["text.primary"]}; }}
QLabel[role="lockText"] {{ font-weight: 600; }}
QFrame[metric="true"] {{
    background: {t["surface.bg"]}; border: 1px solid {t["border"]}; border-radius: {rc}px;
}}

/* Status pills (three separate statuses, GUI_DESIGN.md, 2.4) */
QLabel[status] {{ font-size: 9pt; font-weight: 600; padding: 2px 0; }}
QLabel[status="success"] {{ color: {t["status.success"]}; }}
QLabel[status="warning"] {{ color: {t["status.warning"]}; }}
QLabel[status="error"] {{ color: {t["status.error"]}; }}
QLabel[status="neutral"] {{ color: {t["status.neutral"]}; }}

/* Input fields */
QLineEdit, QComboBox {{
    background: {t["surface.bg"]}; color: {t["text.primary"]};
    border: 1px solid {t["border.strong"]}; border-radius: {rc}px;
    min-height: {h - 2}px; padding: 0 8px;
    selection-background-color: {t["accent"]}; selection-color: {t["accent.fg"]};
}}
QLineEdit:focus, QComboBox:focus {{ border: 1px solid {t["focus.ring"]}; }}
QLineEdit:disabled, QComboBox:disabled {{ color: {t["text.disabled"]}; background: {t["surface.alt"]}; }}
QComboBox::drop-down {{ border: none; width: 22px; }}
QComboBox::down-arrow {{ image: url("{chevron}"); width: 12px; height: 12px; }}
QComboBox QAbstractItemView {{
    background: {t["surface.bg"]}; border: 1px solid {t["border.strong"]};
    selection-background-color: {t["accent.soft"]}; selection-color: {t["text.primary"]};
}}
QFrame#UnitField {{
    background: {t["surface.bg"]};
    border: 1px solid {t["border.strong"]}; border-radius: {rc}px;
    min-height: {h - 2}px; max-height: {h - 2}px;
}}
QFrame#UnitField[focused="true"] {{ border: 1px solid {t["focus.ring"]}; }}
QFrame#UnitField[invalid="true"] {{ border: 1px solid {t["status.error"]}; }}
QFrame#UnitField[locked="true"] {{ background: {t["surface.alt"]}; border-color: {t["border"]}; }}
QLineEdit:read-only {{ color: {t["text.secondary"]}; }}
QLabel[role="fieldError"] {{ color: {t["status.error"]}; font-size: 8.5pt; }}
QFrame#UnitField QLineEdit {{ border: none; background: transparent; padding: 0 8px; }}
QFrame#UnitField QLabel {{ color: {t["text.secondary"]}; font-size: 9pt; padding-right: 8px; }}
QFrame#UnitField QComboBox {{
    border: none; border-left: 1px solid {t["border"]}; border-radius: 0;
    background: transparent; color: {t["text.secondary"]}; min-width: 44px; padding: 0 4px 0 8px;
}}

/* Buttons */
QPushButton {{
    background: {t["surface.bg"]}; color: {t["text.primary"]};
    border: 1px solid {t["border.strong"]}; border-radius: {rc}px;
    min-height: {h - 2}px; padding: 0 14px;
}}
QPushButton:hover {{ background: {t["surface.alt"]}; }}
QPushButton:focus {{ border: 1px solid {t["focus.ring"]}; }}
QPushButton:disabled {{ color: {t["text.disabled"]}; border-color: {t["border"]}; }}
QPushButton[variant="primary"] {{
    background: {t["accent"]}; color: {t["accent.fg"]}; border: 1px solid {t["accent"]};
    font-weight: 600;
}}
QPushButton[variant="primary"]:hover {{ background: {t["accent.hover"]}; }}
QPushButton[variant="primary"]:disabled {{
    background: {t["surface.alt"]}; color: {t["text.disabled"]}; border-color: {t["border"]};
}}
QPushButton[variant="danger"] {{ color: {t["status.error"]}; border-color: {t["status.error"]}; }}
QPushButton[segment] {{
    border-radius: 0; min-width: 52px; padding: 0 12px; margin: 0;
}}
QPushButton[segment="first"] {{
    border-top-left-radius: {rc}px; border-bottom-left-radius: {rc}px;
}}
QPushButton[segment="last"] {{
    border-top-right-radius: {rc}px; border-bottom-right-radius: {rc}px;
}}
QPushButton[segment]:checked {{
    background: {t["accent.soft"]}; color: {t["accent"]};
    border-color: {t["accent"]}; font-weight: 600;
}}
QPushButton[segment][strong="true"]:checked {{
    background: {t["accent"]}; color: {t["accent.fg"]}; border-color: {t["accent"]};
}}
QPushButton[segment]:focus, QPushButton[segment][strong="true"]:focus {{
    border: 2px solid {t["focus.ring"]};
}}

/* Gear menu in the header */
QMenu {{
    background: {t["surface.bg"]}; color: {t["text.primary"]};
    border: 1px solid {t["border.strong"]}; border-radius: {rc}px; padding: 4px;
}}
QMenu::item {{ padding: 6px 28px 6px 12px; border-radius: 3px; }}
QMenu::item:selected {{ background: {t["accent.soft"]}; color: {t["text.primary"]}; }}
QMenu::separator {{ height: 1px; background: {t["border"]}; margin: 4px 6px; }}
QFrame#AppHeader QToolButton::menu-indicator {{ image: none; width: 0; }}
QRadioButton {{ spacing: 8px; background: transparent; }}

/* Tables */
QTableWidget {{
    background: {t["surface.bg"]}; alternate-background-color: {t["surface.alt"]};
    border: none; gridline-color: {t["border"]};
    selection-background-color: {t["accent.soft"]}; selection-color: {t["text.primary"]};
}}
QHeaderView::section {{
    background: {t["surface.bg"]}; color: {t["text.secondary"]};
    border: none; border-bottom: 1px solid {t["border"]};
    padding: 6px 8px; font-size: 8.5pt; font-weight: 600;
}}
QTableCornerButton::section {{ background: {t["surface.bg"]}; border: none; }}

/* Analysis tabs (horizontal, top) */
QTabWidget::pane {{ border: none; border-top: 1px solid {t["border"]}; top: -1px; }}
QTabBar {{ background: transparent; }}
QTabBar::tab {{
    background: transparent; color: {t["text.secondary"]};
    padding: 9px 16px; border: none; border-bottom: 2px solid transparent;
    font-size: 9.5pt;
}}
QTabBar::tab:selected {{ color: {t["accent"]}; border-bottom: 2px solid {t["accent"]}; font-weight: 600; }}
QTabBar::tab:hover:!selected {{ color: {t["text.primary"]}; }}
QTabBar::tab:disabled {{ color: {t["text.disabled"]}; }}
QFrame#TabToolbar, QFrame#StatusStrip, QFrame#ActionBar {{
    background: {t["surface.bg"]}; border: none; border-bottom: 1px solid {t["border"]};
}}
QFrame#ActionBar {{ border-bottom: none; border-top: 1px solid {t["border"]}; }}
QFrame#TabToolbar QLabel, QFrame#StatusStrip QLabel, QFrame#ActionBar QLabel {{ background: transparent; }}

/* Figure placeholder (screen figures follow the theme; export does not) */
QFrame[figure="true"] {{
    background: {t["plot.bg"]}; border: 1px solid {t["border"]}; border-radius: {rk}px;
}}
QFrame[figure="true"] QLabel {{ background: transparent; border: none; }}
QFrame[figure="true"] QToolButton {{
    background: transparent; color: {t["text.secondary"]};
    border: 1px solid {t["border"]}; border-radius: {rc}px; padding: 2px 7px; font-size: 8.5pt;
}}
QFrame[figure="true"] QToolButton:disabled {{ color: {t["text.disabled"]}; }}
QLabel[role="figureTitle"] {{ font-size: 10pt; font-weight: 600; }}
QFrame#FigureArea {{
    background: transparent; border: 1px dashed {t["plot.axis"]}; border-radius: {rc}px;
}}
QLabel[role="figureMark"], QFrame[figure="true"] QLabel[role="figureMark"] {{
    color: {t["testdata.fg"]}; background: {t["testdata.bg"]};
    border: 1px solid {t["testdata.border"]}; border-radius: 3px;
    padding: 1px 6px; font-size: 8pt; font-weight: 600;
}}

/* Inspector, scroll areas, status bar */
QFrame#Inspector {{ background: {t["surface.bg"]}; border: none; border-left: 1px solid {t["border"]}; }}
QFrame#Inspector QLabel {{ background: transparent; }}
QScrollArea {{ background: transparent; border: none; }}
QScrollArea > QWidget > QWidget {{ background: transparent; }}
QStatusBar {{
    background: {t["surface.bg"]}; color: {t["text.secondary"]};
    border-top: 1px solid {t["border"]}; font-size: 8.5pt;
}}
QStatusBar QLabel {{ color: {t["text.secondary"]}; padding: 0 6px; }}
QStatusBar QComboBox {{ min-height: 20px; font-size: 8.5pt; }}
"""
