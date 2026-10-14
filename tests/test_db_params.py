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




class _RecordingCursor:
    def __init__(self, statements):
        self.statements = statements

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, statement, *args):
        self.statements.append(str(statement))


class _RecordingConn:
    def __init__(self):
        self.statements = []

    def cursor(self):
        return _RecordingCursor(self.statements)

    def commit(self):
        pass


def test_init_db_v2_does_not_require_italian_config_by_default(monkeypatch):
    from rag_indexer import db as db_module

    monkeypatch.setattr(db_module, "_register_vector_safe", lambda conn: None)
    monkeypatch.setattr(db_module.sql, "SQL", lambda text: type("S", (), {"format": lambda self, **kw: text})())
    conn = _RecordingConn()
    db_module.init_db_v2(conn, 4, 1)

    assert not any("italian" in statement for statement in conn.statements)
    assert not any("unaccent" in statement for statement in conn.statements)


def test_init_db_v2_creates_italian_unaccent_on_request(monkeypatch):
    from rag_indexer import db as db_module

    monkeypatch.setattr(db_module, "_register_vector_safe", lambda conn: None)
    monkeypatch.setattr(db_module.sql, "SQL", lambda text: type("S", (), {"format": lambda self, **kw: text})())
    conn = _RecordingConn()
    db_module.init_db_v2(conn, 4, 1, fts_config="italian_unaccent")

    assert any("CREATE TEXT SEARCH CONFIGURATION italian_unaccent" in statement for statement in conn.statements)
