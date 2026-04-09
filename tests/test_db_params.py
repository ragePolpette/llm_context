import pytest

from rag_indexer import db as db_module


class _FakeCursor:
    def __init__(self):
        self.executed = None
        self.executemany_calls = None

    def execute(self, query, values):
        self.executed = (str(query), values)

    def executemany(self, query, values):
        self.executemany_calls = (str(query), list(values))


def test_normalize_params_reorders_values_by_placeholder_index():
    query, order = db_module._normalize_params(
        "SELECT * FROM table WHERE a=@Valore1 AND b=@Valore0 AND c=@Valore1"
    )

    assert query == "SELECT * FROM table WHERE a=%s AND b=%s AND c=%s"
    assert order == [1, 0, 1]
    assert db_module._reorder_params(order, ["first", "second"]) == [
        "second",
        "first",
        "second",
    ]


def test_normalize_params_rejects_non_contiguous_placeholders():
    with pytest.raises(ValueError, match="contiguous"):
        db_module._normalize_params("SELECT * FROM table WHERE a=@Valore0 AND b=@Valore2")


def test_execute_params_rejects_wrong_parameter_count():
    cursor = _FakeCursor()

    with pytest.raises(ValueError, match="parameter count mismatch"):
        db_module.execute_params(
            cursor,
            "SELECT * FROM table WHERE a=@Valore0 AND b=@Valore1",
            ["only-one"],
        )


def test_execute_many_params_reorders_each_row():
    cursor = _FakeCursor()

    db_module.execute_many_params(
        cursor,
        "INSERT INTO t(a, b) VALUES (@Valore1, @Valore0)",
        [["first", "second"], ["left", "right"]],
    )

    assert "INSERT INTO t(a, b) VALUES (%s, %s)" in cursor.executemany_calls[0]
    assert cursor.executemany_calls[1] == [
        ["second", "first"],
        ["right", "left"],
    ]


