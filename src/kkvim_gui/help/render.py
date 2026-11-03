"""Article -> Markdown for the help viewer. Qt-free.

Reserved notation (GUI_DESIGN.md, 17.4), rendered simply for now:
- `$$ ... $$` — display formula in LaTeX: shown as a monospace block with a note;
- `$ ... $`   — inline formula: shown as monospace text;
- `[@KEY, с. N, ур. (k)]` — reference to a source: shown in brackets, linked to the
  literature section.
Graphical formula rendering is a later stage. No formulas are written by the GUI.
"""

import re

from kkvim_gui.help.articles import CATEGORIES, STATUSES, get

FORMULA_BLOCK = re.compile(r"\$\$(.+?)\$\$", re.S)
INLINE_FORMULA = re.compile(r"(?<!\$)\$([^$\n]+?)\$(?!\$)")
CITATION = re.compile(r"\[@([^\]]+)\]")
FORMULA_NOTE = "Формула в записи LaTeX; графическое отображение формул — на следующих этапах."
STUB_NOTE = (
    "Раздел подготовлен как заготовка. Научное содержание готовится в основном проекте "
    "по проверенным источникам и будет добавлено позднее."
)
LITERATURE_TOPIC = "literature.overview"


def render_body(markdown):
    markdown = FORMULA_BLOCK.sub(
        lambda m: f"\n```\n{m.group(1).strip()}\n```\n\n*{FORMULA_NOTE}*\n", markdown
    )
    markdown = INLINE_FORMULA.sub(lambda m: f"`{m.group(1).strip()}`", markdown)
    return CITATION.sub(lambda m: f"[\\[{m.group(1).strip()}\\]](topic:{LITERATURE_TOPIC})", markdown)


def article_markdown(article):
    parts = [
        f"# {article.title}",
        f"*{CATEGORIES[article.category]} · {STATUSES[article.status]}*",
        f"**{article.summary}**",
    ]
    if article.body:
        parts.append(render_body(article.body))
    if article.status == "stub":
        parts.append(f"> {STUB_NOTE}")
    related = [get(topic) for topic in article.related]
    if any(related):
        parts.append("## Связанные темы")
        parts.append("\n".join(f"- [{a.title}](topic:{a.id})" for a in related if a))
    if article.sources:
        parts.append("## Основание")
        parts.append("\n".join(f"- {s}" for s in article.sources))
    return "\n\n".join(parts) + "\n"
