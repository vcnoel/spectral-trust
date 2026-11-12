"""mkdocs hook: render the Sphinx roles of the docstrings (``:func:`x```,
``:class:`~a.b.C```) as plain code, the way Sphinx would show them."""
import re

_ROLE = re.compile(r":(?:func|meth|class|mod|data|attr|obj):<code>(~?)([^<]*)</code>")


def _shorten(match: re.Match) -> str:
    tilde, target = match.groups()
    return f"<code>{target.rsplit('.', 1)[-1] if tilde else target}</code>"


def on_page_content(html, **kwargs):
    return _ROLE.sub(_shorten, html)
