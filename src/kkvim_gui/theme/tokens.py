"""Colour and size tokens (GUI_DESIGN.md, sections 10.2 and 10.3).

Both themes define exactly the same keys. Widgets never hard-code colours: they take
them from these tokens through the generated stylesheet or Session.tokens. The
publication export (stage P6) does not use these tokens at all.
"""

LIGHT = {
    "header.bg": "#17324D",
    "header.fg": "#FFFFFF",
    "header.fg.muted": "#B9C7D6",
    "header.hover": "#23476A",
    "window.bg": "#F3F6F9",
    "surface.bg": "#FFFFFF",
    "surface.alt": "#EEF3F7",
    "border": "#DCE3EA",
    "border.strong": "#C3CDD7",
    "text.primary": "#1B2B3A",
    "text.secondary": "#5F6E7C",
    "text.disabled": "#9AA6B2",
    "accent": "#0B5C8A",
    "accent.hover": "#094C72",
    "accent.fg": "#FFFFFF",
    "accent.soft": "#E3EEF6",
    "focus.ring": "#2F86BD",
    "status.success": "#1F7A5A",
    "status.warning": "#B7791F",
    "status.error": "#B23B3B",
    "status.neutral": "#5F6E7C",
    "testdata.bg": "#FFF4D6",
    "testdata.fg": "#7A4B00",
    "testdata.border": "#E8C46A",
    "plot.bg": "#FFFFFF",
    "plot.grid": "#E6EBF0",
    "plot.axis": "#5F6E7C",
    "icon.chevron": "chevron-down-light.svg",  # stroke = text.secondary
}

DARK = {
    "header.bg": "#10263B",
    "header.fg": "#E8EEF4",
    "header.fg.muted": "#93A5B8",
    "header.hover": "#1B3A57",
    "window.bg": "#12171D",
    "surface.bg": "#1A212A",
    "surface.alt": "#212B36",
    "border": "#2E3A47",
    "border.strong": "#3D4B5A",
    "text.primary": "#E4EAF0",
    "text.secondary": "#9AA9B7",
    "text.disabled": "#5E6B78",
    "accent": "#2F7DB3",
    "accent.hover": "#3E8FC4",
    "accent.fg": "#FFFFFF",
    "accent.soft": "#1E3346",
    "focus.ring": "#6CB4E4",
    "status.success": "#4CC08F",
    "status.warning": "#E3AE52",
    "status.error": "#E57373",
    "status.neutral": "#9AA9B7",
    "testdata.bg": "#3A2E12",
    "testdata.fg": "#F2C766",
    "testdata.border": "#8A6A24",
    "plot.bg": "#161C23",
    "plot.grid": "#27313C",
    "plot.axis": "#9AA9B7",
    "icon.chevron": "chevron-down-dark.svg",  # stroke = text.secondary
}

THEMES = {"light": LIGHT, "dark": DARK}
DEFAULT_THEME = "light"

# Sizes in Qt logical pixels (device-independent; scaled by Qt on HiDPI screens).
SIZES = {
    "header.height": 46,
    "control.height": 32,
    "card.padding": 14,
    "gap": 12,
    "radius.card": 6,
    "radius.control": 4,
    "inspector.width": 280,
}
