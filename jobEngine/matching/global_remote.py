"""Detects "global remote" jobs: remote roles open to candidates anywhere,
posted outside India. Used by the normalization node to add the
``GLOBAL_REMOTE_TAG`` tag; jobs located in India never get it.

Two kinds of description phrases:
  * strong — "remote from anywhere", "work from anywhere", "anywhere in the world",
    "global remote", "location independent", ... — always count.
  * weak   — "fully remote", "100% remote", "remote-first" — count only when the
    text doesn't restrict where the candidate must live ("US only", "must be
    located in", "authorized to work in", ...).
Any text limiting the role to India ("anywhere in India", "pan India") cancels both.
"""

from __future__ import annotations

import re

GLOBAL_REMOTE_TAG = "Global Remote"

_FLAGS = re.IGNORECASE | re.DOTALL

_STRONG_RE = re.compile(
    "|".join(
        [
            r"\bglobal(?:ly)?[\s-]+remote\b",
            r"\bremote[\s-]*[(\[]?\s*(?:global|worldwide|anywhere)\b",
            r"\b(?:remote|work|working|remotely)\s+(?:from\s+)?anywhere\b",
            r"\banywhere\s+in\s+the\s+world\b",
            r"\blocation[\s-]+(?:independent|agnostic)\b",
            r"\bfrom\s+any\s+(?:country|location)\b",
            r"\b(?:candidates|applicants|talent)\s+(?:from\s+)?(?:anywhere|worldwide|around\s+the\s+world)\b",
            r"\bworldwide[\s,-]+remote\b",
            r"\b(?:hire|hiring|recruiting|open)\s+(?:remotely\s+)?(?:worldwide|globally)\b",
        ]
    ),
    _FLAGS,
)
_WEAK_RE = re.compile(
    r"\b(?:fully|100\s*%|completely|entirely|totally)[\s-]+remote\b|\bremote[\s-]+first\b",
    _FLAGS,
)
_REGIONS = r"(?:us|usa|u\.s\.?|united\s+states|uk|u\.k\.?|united\s+kingdom|eu|europe|emea|latam|apac|canada|australia|germany)"
_RESTRICTED_RE = re.compile(
    "|".join(
        [
            rf"\b{_REGIONS}[\s-]+(?:only|based\s+(?:candidates|applicants|only))\b",
            rf"\b(?:within|in)\s+(?:the\s+)?{_REGIONS}\s+only\b",
            r"\bmust\s+(?:be\s+)?(?:located|based|residing|reside|live|living)\s+in\b",
            r"\b(?:authori[sz]ed|eligible|legally\s+able)\s+to\s+work\s+in\b",
            r"\b(?:only|exclusively)\s+(?:open\s+to|hiring|accepting|considering)\s+(?:candidates|applicants)\s+(?:in|from|based)\b",
            r"\bresidents?\s+of\b",
        ]
    ),
    _FLAGS,
)
_INDIA_ONLY_RE = re.compile(
    r"\b(?:anywhere|remote|remotely|within|across)\s+(?:in\s+)?india\b|\bpan[\s-]+india\b|\bindia[\s-]+(?:only|based)\b",
    _FLAGS,
)
# Location strings: LinkedIn shows "Bengaluru, Karnataka, India" / "India (Remote)".
_INDIA_LOCATION_RE = re.compile(
    r"\b(?:india|bharat|bengaluru|bangalore|mumbai|delhi|ncr|noida|gurgaon|gurugram|hyderabad|"
    r"chennai|pune|kolkata|ahmedabad|kochi|cochin|jaipur|chandigarh|indore|coimbatore|"
    r"thiruvananthapuram|trivandrum|mysuru|mysore|nagpur|lucknow|bhubaneswar|visakhapatnam|"
    r"karnataka|maharashtra|telangana|tamil\s+nadu|kerala|haryana|uttar\s+pradesh|"
    r"west\s+bengal|gujarat|rajasthan|andhra\s+pradesh|odisha|punjab)\b",
    re.IGNORECASE,
)
_GLOBAL_LOCATION_RE = re.compile(r"\b(?:worldwide|anywhere|global)\b", re.IGNORECASE)


def is_india_location(location: str | None) -> bool:
    return bool(_INDIA_LOCATION_RE.search(location or ""))


def is_global_remote(location: str | None, description: str | None) -> bool:
    """True for a job outside India whose location or description says it can be
    done from anywhere."""
    if is_india_location(location):
        return False
    text = description or ""
    if _INDIA_ONLY_RE.search(text):
        return False
    if _GLOBAL_LOCATION_RE.search(location or "") or _STRONG_RE.search(text):
        return True
    return bool(_WEAK_RE.search(text)) and not _RESTRICTED_RE.search(text)
