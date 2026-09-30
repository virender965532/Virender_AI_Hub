"""Read / replace the ``keywords`` query parameter of a source's search URL.

A search request's ``search_keywords`` are used only here: they replace the
``keywords=`` value of each configured ``search_urls`` entry that has one
(e.g. LinkedIn). They are not used as a filter anywhere else. URLs without a
``keywords`` parameter are left unchanged.
"""

from __future__ import annotations

from urllib.parse import parse_qsl, quote, urlencode, urlsplit, urlunsplit

KEYWORDS_PARAM = "keywords"


def url_keywords(url: str) -> str | None:
    for key, value in parse_qsl(urlsplit(url).query, keep_blank_values=True):
        if key == KEYWORDS_PARAM:
            return value
    return None


def with_keywords(url: str, keywords: str) -> str:
    parts = urlsplit(url)
    params = parse_qsl(parts.query, keep_blank_values=True)
    if not any(key == KEYWORDS_PARAM for key, _ in params):
        return url
    params = [(key, keywords if key == KEYWORDS_PARAM else value) for key, value in params]
    return urlunsplit(parts._replace(query=urlencode(params, quote_via=quote)))
