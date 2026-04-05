"""Tests for canonical mode ordering."""

from operator_algebra.algebra import _mode_sort_key


def test_int_ordering():
    assert _mode_sort_key(1) < _mode_sort_key(2)


def test_str_ordering():
    assert _mode_sort_key("a") < _mode_sort_key("b")


def test_tuple_mode():
    key = _mode_sort_key((1, "up"))
    assert isinstance(key, tuple)


def test_mixed_types_comparable():
    k1 = _mode_sort_key(1)
    k2 = _mode_sort_key("up")
    # Should not raise
    assert (k1 < k2) or (k1 >= k2)


def test_bool_int_aliasing():
    assert _mode_sort_key(True) == _mode_sort_key(1)
