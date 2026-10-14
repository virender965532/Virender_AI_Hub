"""LinkedIn — auth-gated source (section 10).

LinkedIn serves two different search pages, each with its own adapter:

  * ``/jobs/search/``         → ``LinkedInSearchAdapter``        (``search_adapter.py``)
  * ``/jobs/search-results/`` → ``LinkedInSearchResultsAdapter`` (``search_results_adapter.py``)

``adapter: linkedin`` picks one of them per search URL, so a single source can
mix both kinds of URL in ``search_urls``. Session handling, sign-in and
pagination are shared and live in ``base.py``.
"""

from __future__ import annotations

from urllib.parse import urlparse

from playwright.async_api import Page

from ...models.job import RawJob
from ..registry import register_adapter
from .base import _DEFAULT_SIGNIN_URL, _is_checkpoint_url, _is_login_url, LinkedInBaseAdapter  # noqa: F401
from .search_adapter import LinkedInSearchAdapter
from .search_results_adapter import LinkedInSearchResultsAdapter


def _is_search_page(url: str) -> bool:
    """True for the classic ``/jobs/search/`` page, False for ``/jobs/search-results/``
    (and anything else, which the search-results adapter has always handled)."""
    return urlparse(url).path.rstrip("/").endswith("/jobs/search")


@register_adapter("linkedin")
class LinkedInAdapter(LinkedInBaseAdapter):
    def __init__(self) -> None:
        self._search = LinkedInSearchAdapter()
        self._search_results = LinkedInSearchResultsAdapter()

    async def _extract_page(
        self, page: Page, *, cfg, source_url: str, seen_ids: set[str], limit: int,
        known: dict[str, str] | None = None,
    ) -> tuple[list[RawJob], int]:
        adapter = self._search if _is_search_page(source_url) else self._search_results
        return await adapter._extract_page(
            page, cfg=cfg, source_url=source_url, seen_ids=seen_ids, limit=limit, known=known
        )
