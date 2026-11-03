"""Help centre window (GUI_DESIGN.md, 17.3): search on top, alphabetical index А–Я, article view."""

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QFrame,
    QLabel,
    QLineEdit,
    QMainWindow,
    QSplitter,
    QTextBrowser,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from kkvim_gui.help.articles import alphabetical_index, get, search
from kkvim_gui.help.render import article_markdown
from kkvim_gui.widgets.common import fit_to_screen
from kkvim_gui.widgets.header import AppHeader, app_icon

START_TOPIC = "help.using"
TOPIC_ROLE = Qt.ItemDataRole.UserRole


class HelpWindow(QMainWindow):
    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.setWindowTitle("КК-ВИМ — справка")
        self.setWindowIcon(app_icon())
        self.current_topic = None
        self.mode = "topic"  # "topic": opened for an article; "search": general search

        central = QWidget()
        central.setObjectName("Page")
        lay = QVBoxLayout(central)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(AppHeader(session, "КК-ВИМ: справка", show_badge=False,
                                show_settings=False, show_help=False))

        side = QFrame()
        side.setObjectName("HelpIndex")
        side_lay = QVBoxLayout(side)
        side_lay.setContentsMargins(12, 12, 12, 12)
        side_lay.setSpacing(8)
        self.search = QLineEdit()
        self.search.setPlaceholderText("Поиск по справке")
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self._fill_index)
        self.found = QLabel("")
        self.found.setProperty("role", "secondary")
        self.index = QTreeWidget()
        self.index.setHeaderHidden(True)
        self.index.setRootIsDecorated(False)
        self.index.setIndentation(12)
        self.index.itemClicked.connect(self._on_item)
        self.index.itemActivated.connect(self._on_item)
        side_lay.addWidget(self.search)
        side_lay.addWidget(self.found)
        side_lay.addWidget(self.index, 1)

        self.view = QTextBrowser()
        self.view.setObjectName("HelpArticle")
        self.view.setOpenLinks(False)
        self.view.anchorClicked.connect(self._on_link)
        self.view.document().setDocumentMargin(18)

        splitter = QSplitter()
        splitter.addWidget(side)
        splitter.addWidget(self.view)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([300, 700])
        lay.addWidget(splitter, 1)
        self.setCentralWidget(central)

        self._fill_index("")
        fit_to_screen(self, design=(1100, 760), minimum=(720, 480))

    def _fill_index(self, query):
        """Alphabetical index А–Я; with a query, only the matching articles."""
        self.index.clear()
        matches = {a.id for a in search(query)} if query.strip() else None
        bold = QFont(self.index.font())
        bold.setBold(True)
        count = 0
        for letter, articles in alphabetical_index():
            shown = [a for a in articles if matches is None or a.id in matches]
            if not shown:
                continue
            head = QTreeWidgetItem([letter])
            head.setFont(0, bold)
            head.setFlags(Qt.ItemFlag.ItemIsEnabled)
            self.index.addTopLevelItem(head)
            for article in shown:
                item = QTreeWidgetItem([article.title])
                item.setData(0, TOPIC_ROLE, article.id)
                item.setToolTip(0, article.summary)
                head.addChild(item)
                count += 1
        self.index.expandAll()
        self.found.setText(f"Найдено: {count}" if matches is not None else f"Статей: {count}")
        self._select(self.current_topic)

    def _select(self, topic):
        for i in range(self.index.topLevelItemCount()):
            head = self.index.topLevelItem(i)
            for j in range(head.childCount()):
                item = head.child(j)
                if item.data(0, TOPIC_ROLE) == topic:
                    self.index.setCurrentItem(item)
                    return

    def topics_in_index(self):
        return [
            self.index.topLevelItem(i).child(j).data(0, TOPIC_ROLE)
            for i in range(self.index.topLevelItemCount())
            for j in range(self.index.topLevelItem(i).childCount())
        ]

    def _on_item(self, item, _column=0):
        topic = item.data(0, TOPIC_ROLE)
        if topic:
            self.show_topic(topic)

    def _on_link(self, url):
        if url.scheme() == "topic":
            self.show_topic(url.path())

    def show_topic(self, topic):
        article = get(topic)
        if article is None:
            self.show_search()
            return
        self.current_topic = topic
        self.mode = "topic"
        self.view.setMarkdown(article_markdown(article))
        self._select(topic)

    def show_search(self, query=""):
        self.search.setText(query)
        self.show_topic(START_TOPIC)
        self.mode = "search"
        self.search.setFocus()
        self.search.selectAll()
