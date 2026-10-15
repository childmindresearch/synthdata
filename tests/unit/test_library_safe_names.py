"""Unit tests for synthdata.utils.library_safe_names."""

import pytest

from synthdata.utils import library_safe_names

pytestmark = pytest.mark.unit


def test_unsafe_characters_get_readable_stand_ins():
    assert library_safe_names(["BP [mmHg]", "a,b", "ward: name", "x<5", "ok"]) == {
        "BP [mmHg]": "BP (mmHg)",
        "a,b": "a;b",
        "ward: name": "ward - name",
        "x<5": "xlt5",
    }


def test_collisions_get_a_suffix_so_the_mapping_stays_one_to_one():
    renames = library_safe_names(["a(1)", "a[1]", "a{1}"])
    assert renames == {"a[1]": "a(1)__2", "a{1}": "a(1)__3"}
    assert len(set(renames.values()) | {"a(1)"}) == 3


def test_safe_names_need_no_change():
    assert library_safe_names(["age", "Social/Family Functioning", 3]) == {}
