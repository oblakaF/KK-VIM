"""Help articles: one Markdown file per topic with a stable topic id. Qt-free.

File `articles/<topic id>.md`:

    ---
    id: param.geometry.a
    title: Длина пластины a
    category: model
    summary: One or two sentences: the tooltip text and the lead of the article.
    keywords: comma, separated, words
    related: param.geometry.b, param.geometry.h
    sources: semicolon; separated; references
    status: stub
    ---
    Markdown body

The tooltip and the article share the `summary` field, so the short and the full
definitions cannot diverge. Topic ids are permanent: GUI widgets and links refer to
them. Scientific articles are written in the main project from verified sources and
added later; this package does not generate them (GUI_DESIGN.md, 17.4).
"""

import re
from dataclasses import dataclass
from functools import cache
from importlib.resources import files

# Thematic categories: metadata for future filters; the main index is alphabetical.
CATEGORIES = {
    "start": "Начало работы",
    "model": "Параметры модели",
    "theories": "Теории пластин",
    "method": "Метод КК-ВИМ",
    "bc": "Граничные условия",
    "iterations": "Итерации и сходимость",
    "results": "Результаты",
    "verification": "Независимые методы и эталоны",
    "literature": "Научная литература",
    "limitations": "Ограничения программы",
}

STATUSES = {
    "stub": "Заготовка раздела",
    "draft": "Черновик",
    "reviewed": "Проверено",
}

TOPIC_ID = re.compile(r"^[a-z][a-z0-9_]*(\.[A-Za-z0-9_]+)+$")
SUMMARY_MAX = 240
REQUIRED = ("id", "title", "category", "summary", "status")


@dataclass(frozen=True)
class Article:
    id: str
    title: str
    category: str
    summary: str
    status: str
    body: str = ""
    keywords: tuple = ()
    related: tuple = ()
    sources: tuple = ()


def _split(value, sep):
    return tuple(part.strip() for part in value.split(sep) if part.strip())


def parse_article(text, name="<text>"):
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValueError(f"{name}: the file must start with a '---' header")
    try:
        end = lines.index("---", 1)
    except ValueError:
        raise ValueError(f"{name}: the header is not closed with '---'") from None
    meta = {}
    for line in lines[1:end]:
        if not line.strip():
            continue
        key, sep, value = line.partition(":")
        if not sep:
            raise ValueError(f"{name}: header line without ':': {line!r}")
        meta[key.strip()] = value.strip()
    missing = [k for k in REQUIRED if not meta.get(k)]
    if missing:
        raise ValueError(f"{name}: missing header fields {missing}")
    if not TOPIC_ID.match(meta["id"]):
        raise ValueError(f"{name}: invalid topic id {meta['id']!r}")
    if meta["category"] not in CATEGORIES:
        raise ValueError(f"{name}: unknown category {meta['category']!r}")
    if meta["status"] not in STATUSES:
        raise ValueError(f"{name}: unknown status {meta['status']!r}")
    if len(meta["summary"]) > SUMMARY_MAX:
        raise ValueError(f"{name}: summary longer than {SUMMARY_MAX} characters")
    return Article(
        id=meta["id"],
        title=meta["title"],
        category=meta["category"],
        summary=meta["summary"],
        status=meta["status"],
        body="\n".join(lines[end + 1:]).strip(),
        keywords=_split(meta.get("keywords", ""), ","),
        related=_split(meta.get("related", ""), ","),
        sources=_split(meta.get("sources", ""), ";"),
    )


@cache
def load_articles():
    """All articles by topic id. The file name must equal the topic id."""
    root = files("kkvim_gui.help").joinpath("articles")
    articles = {}
    for entry in sorted(root.iterdir(), key=lambda e: e.name):
        if not entry.name.endswith(".md"):
            continue
        article = parse_article(entry.read_text("utf-8"), entry.name)
        if entry.name[:-3] != article.id:
            raise ValueError(f"{entry.name}: file name differs from topic id {article.id!r}")
        if article.id in articles:
            raise ValueError(f"duplicate topic id {article.id!r}")
        articles[article.id] = article
    return articles


def get(topic_id):
    return load_articles().get(topic_id) if topic_id else None


def normalize(text):
    """Case- and ё-insensitive form used for sorting and search."""
    return text.casefold().replace("ё", "е")


def sort_key(article):
    return normalize(article.title)


def alphabetical_index():
    """[(letter, [articles sorted by title])], letters in А-Я order (then others)."""
    groups = {}
    for article in sorted(load_articles().values(), key=sort_key):
        first = article.title[:1].upper().replace("Ё", "Е")
        groups.setdefault(first, []).append(article)

    def letter_key(letter):
        cyrillic = "А" <= letter <= "Я"
        return (0 if cyrillic else 1, normalize(letter))

    return [(letter, groups[letter]) for letter in sorted(groups, key=letter_key)]


def search(query):
    """Articles whose title, summary, keywords or body contain every query word."""
    words = normalize(query).split()
    if not words:
        return sorted(load_articles().values(), key=sort_key)
    found = []
    for article in load_articles().values():
        haystack = normalize(" ".join((article.title, article.summary, " ".join(article.keywords), article.body)))
        if all(word in haystack for word in words):
            found.append(article)
    return sorted(found, key=sort_key)
