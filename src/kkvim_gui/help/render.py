"""Article -> Markdown for the help viewer. Qt-free.

Reserved notation (GUI_DESIGN.md, 17.4), rendered simply for now:
- `$$ ... $$` — display formula in LaTeX: shown as a monospace block with a note;
- `$ ... $`   — inline formula: shown as monospace text;
- `[@KEY, с. N, ур. (k)]` — reference to a source: shown in brackets, linked to the
  literature section.
Graphical formula rendering is a later stage. No formulas are written by the GUI.

What the user sees (GUI-PLATE-WORKFLOW-UX-01): the title, the short definition (the same text as the tooltip), the
article, related topics and, where the article rests on scientific literature, «Источники и цитирование» (the
``citations`` header field: author, title, year, bibliographic data, page and formula). The editorial status of the
article and its project basis (``status``, ``sources`` — paths of project documents and decision numbers) are metadata
for developers and are not shown.
"""

import re

from kkvim_gui.help.articles import get

FORMULA_BLOCK = re.compile(r"\$\$(.+?)\$\$", re.S)
INLINE_FORMULA = re.compile(r"(?<!\$)\$([^$\n]+?)\$(?!\$)")
CITATION = re.compile(r"\[@([^\]]+)\]")
FORMULA_NOTE = "Формула в записи LaTeX; графическое отображение формул — на следующих этапах."
STUB_NOTE = (  # kept for compatibility; never rendered: help shows no placeholder text
    "Раздел подготовлен как заготовка. Научное содержание готовится в основном проекте "
    "по проверенным источникам и будет добавлено позднее."
)
LITERATURE_TOPIC = "literature.overview"
CITATIONS_TITLE = "Источники и цитирование"


def render_body(markdown):
    markdown = FORMULA_BLOCK.sub(
        lambda m: f"\n```\n{m.group(1).strip()}\n```\n\n*{FORMULA_NOTE}*\n", markdown
    )
    markdown = INLINE_FORMULA.sub(lambda m: f"`{m.group(1).strip()}`", markdown)
    return CITATION.sub(lambda m: f"[\\[{m.group(1).strip()}\\]](topic:{LITERATURE_TOPIC})", markdown)


def article_markdown(article):
    parts = [f"# {article.title}", f"**{article.summary}**"]
    if article.body:
        parts.append(render_body(article.body))
    related = [get(topic) for topic in article.related]
    if any(related):
        parts.append("## Связанные темы")
        parts.append("\n".join(f"- [{a.title}](topic:{a.id})" for a in related if a))
    if article.citations:
        parts.append(f"## {CITATIONS_TITLE}")
        parts.append("\n".join(f"- {c}" for c in article.citations))
    return "\n\n".join(parts) + "\n"
