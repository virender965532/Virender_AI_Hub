from __future__ import annotations

import pytest

from jobEngine.matching.global_remote import is_global_remote, is_india_location


@pytest.mark.parametrize(
    ("location", "description"),
    [
        ("United States (Remote)", "This role is remote from anywhere."),
        ("Germany", "You can work from anywhere in the world."),
        ("European Union (Remote)", "We are a globally remote team."),
        ("Worldwide", "Join our team."),
        ("Poland (Remote)", "The position is fully remote. Apply now."),
        ("Netherlands", "Location independent role, 100% remote."),
        ("United Kingdom", "Remote (Worldwide)"),
    ],
)
def test_global_remote_outside_india(location, description):
    assert is_global_remote(location, description)


@pytest.mark.parametrize(
    ("location", "description"),
    [
        # Located in India: never tagged, whatever the description says.
        ("India (Remote)", "Remote from anywhere."),
        ("Bengaluru, Karnataka, India", "Work from anywhere in the world."),
        ("Noida", "Fully remote."),
        # Outside India but restricted, or not remote-anywhere at all.
        ("United States (Remote)", "Fully remote, US only."),
        ("United States", "100% remote. Must be located in the United States."),
        ("Canada (Remote)", "Fully remote; you must be authorized to work in Canada."),
        ("United States", "Remote anywhere in India."),
        ("Germany", "Hybrid role in our Berlin office."),
        ("United States (Remote)", "Remote role with our worldwide customers in the healthcare sector across many states."),
    ],
)
def test_not_global_remote(location, description):
    assert not is_global_remote(location, description)


def test_is_india_location():
    assert is_india_location("Gurugram, Haryana")
    assert is_india_location("India")
    assert not is_india_location("Indiana, United States")
    assert not is_india_location(None)
