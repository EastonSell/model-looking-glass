"""The full-row-prefix comparison has no access to private splits."""
from tools.audit_natural_successor_public_support import prefix_counts


def test_both_complete_row_prefix_directions_and_exact_matches():
    old = [[1, 2], [3, 4, 5], [7, 8]]
    new = [[1, 2, 9], [3, 4], [7, 8], [9, 8]]
    assert prefix_counts(old, new) == (2, 2)


def test_shared_short_prefix_alone_is_not_a_complete_row_collision():
    assert prefix_counts([[1, 2, 3]], [[1, 2, 4]]) == (0, 0)
