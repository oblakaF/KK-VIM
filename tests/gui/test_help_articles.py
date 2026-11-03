"""Help articles: files, stable topic ids, alphabetical index, search, rendering. Qt-free."""

import pytest

from kkvim_gui.help.articles import (
    CATEGORIES,
    SUMMARY_MAX,
    alphabetical_index,
    get,
    load_articles,
    normalize,
    parse_article,
    search,
)
from kkvim_gui.help.render import FORMULA_NOTE, STUB_NOTE, article_markdown, render_body

REQUIRED_CATEGORIES = (
    "Начало работы",
    "Параметры модели",
    "Теории пластин",
    "Метод КК-ВИМ",
    "Граничные условия",
    "Итерации и сходимость",
    "Результаты",
    "Независимые методы и эталоны",
    "Научная литература",
    "Ограничения программы",
)
GUI_TOPICS = (
    "param.geometry.a", "param.geometry.b", "param.geometry.h",
    "param.material.E", "param.material.nu", "param.theory",
)


def test_articles_load_and_ids_match_file_names():
    articles = load_articles()
    assert articles
    for topic, article in articles.items():
        assert topic == article.id
        assert article.summary and len(article.summary) <= SUMMARY_MAX


def test_required_categories_each_have_an_article():
    assert tuple(CATEGORIES.values()) == REQUIRED_CATEGORIES
    used = {a.category for a in load_articles().values()}
    assert used == set(CATEGORIES)


def test_parameter_topics_used_by_the_gui_exist():
    for topic in GUI_TOPICS:
        assert get(topic) is not None, topic


def test_related_topics_resolve():
    for article in load_articles().values():
        for topic in article.related:
            assert get(topic) is not None, (article.id, topic)


def test_index_is_alphabetical_a_to_ya():
    index = alphabetical_index()
    letters = [letter for letter, _ in index]
    assert all("А" <= letter <= "Я" for letter in letters)
    assert letters == sorted(letters)
    titles = [normalize(a.title) for _, group in index for a in group]
    assert titles == sorted(titles)
    assert len(titles) == len(load_articles())


@pytest.mark.parametrize(("query", "topic"), [
    ("Пуассон", "param.material.nu"),
    ("пуассона", "param.material.nu"),
    ("модуль юнга", "param.material.E"),
    ("ТОЛЩИНА", "param.geometry.h"),
])
def test_search_is_case_insensitive(query, topic):
    assert topic in [a.id for a in search(query)]


def test_search_without_matches_and_empty_query():
    assert search("несуществующееслово") == []
    assert len(search("  ")) == len(load_articles())


def test_short_tooltip_and_article_share_the_definition():
    for article in load_articles().values():
        assert f"**{article.summary}**" in article_markdown(article)


def test_stub_articles_say_that_content_comes_later():
    for article in load_articles().values():
        md = article_markdown(article)
        assert (STUB_NOTE in md) == (article.status == "stub")


def test_reserved_formula_and_citation_notation():
    md = render_body("Текст $x$.\n\n$$\na + b\n$$\n\nСм. [@P82, с. 80, ур. (3)].")
    assert "`x`" in md
    assert "```\na + b\n```" in md
    assert FORMULA_NOTE in md
    assert "(topic:literature.overview)" in md and "P82, с. 80, ур. (3)" in md


def test_shipped_articles_contain_no_formulas():
    """Formulas come only from verified sources of the main project (GUI_DESIGN.md, 17.4)."""
    for article in load_articles().values():
        assert "$" not in article.body, article.id


@pytest.mark.parametrize("text", [
    "no header",
    "---\nid: a.b\n",
    "---\nid: Bad\ntitle: t\ncategory: model\nsummary: s\nstatus: stub\n---\n",
    "---\nid: a.b\ntitle: t\ncategory: nope\nsummary: s\nstatus: stub\n---\n",
    "---\nid: a.b\ntitle: t\ncategory: model\nsummary: s\nstatus: final\n---\n",
    "---\nid: a.b\ntitle: t\ncategory: model\nstatus: stub\n---\n",
])
def test_invalid_articles_are_rejected(text):
    with pytest.raises(ValueError):
        parse_article(text)
