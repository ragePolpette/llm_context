"""
Unit tests for rag_indexer.symbol_extractor.

Tests cover:
- Python extraction (regex fallback always available)
- JavaScript extraction (regex fallback)
- TypeScript extraction (regex fallback)
- C# extraction (regex fallback)
- tree-sitter path (skipped if not installed)
- Fallback behaviour (ImportError → regex)
"""
from __future__ import annotations

import importlib
import sys
from typing import Optional

import pytest

from rag_indexer.symbol_extractor import SymbolInfo, extract_symbols, _extract_symbols_regex


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find(symbols: list[SymbolInfo], name: str, kind: Optional[str] = None) -> Optional[SymbolInfo]:
    for s in symbols:
        if s.name == name and (kind is None or s.kind == kind):
            return s
    return None


# ---------------------------------------------------------------------------
# Python — regex fallback (always available)
# ---------------------------------------------------------------------------

PYTHON_CODE = """\
class MyService:
    def __init__(self):
        pass

    def do_work(self, x):
        return x * 2

async def standalone_func(a, b):
    return a + b
"""


def test_python_regex_class():
    syms = _extract_symbols_regex(PYTHON_CODE, "python")
    s = _find(syms, "MyService", "class")
    assert s is not None
    assert s.line_start == 1


def test_python_regex_method():
    syms = _extract_symbols_regex(PYTHON_CODE, "python")
    s = _find(syms, "do_work", "method")
    assert s is not None
    assert s.line_start > 1


def test_python_regex_function():
    syms = _extract_symbols_regex(PYTHON_CODE, "python")
    s = _find(syms, "standalone_func", "function")
    assert s is not None


def test_python_extract_symbols_delegates():
    """extract_symbols returns results for python (tree-sitter or regex)."""
    syms = extract_symbols(PYTHON_CODE, "python")
    assert any(s.name == "MyService" for s in syms)
    assert any(s.name == "do_work" for s in syms)
    assert any(s.name == "standalone_func" for s in syms)


# ---------------------------------------------------------------------------
# JavaScript — regex fallback
# ---------------------------------------------------------------------------

JS_CODE = """\
export class UserService {
    constructor() {}

    fetchUser(id) {
        return id;
    }
}

export function createUser(name) {
    return { name };
}

const helperFn = async (x) => x + 1;
"""


def test_js_regex_class():
    syms = _extract_symbols_regex(JS_CODE, "javascript")
    s = _find(syms, "UserService", "class")
    assert s is not None


def test_js_regex_function():
    syms = _extract_symbols_regex(JS_CODE, "javascript")
    s = _find(syms, "createUser", "function")
    assert s is not None


def test_js_regex_assigned_arrow():
    syms = _extract_symbols_regex(JS_CODE, "javascript")
    s = _find(syms, "helperFn", "function")
    assert s is not None


def test_js_extract_symbols_delegates():
    syms = extract_symbols(JS_CODE, "javascript")
    assert any(s.name == "UserService" for s in syms)
    assert any(s.name == "createUser" for s in syms)


# ---------------------------------------------------------------------------
# TypeScript — regex fallback
# ---------------------------------------------------------------------------

TS_CODE = """\
export interface IUserRepo {
    find(id: string): Promise<User>;
}

export type UserId = string;

export enum Role {
    Admin,
    User,
}

export class UserRepo implements IUserRepo {
    find(id: string) { return null as any; }
}
"""


def test_ts_regex_interface():
    syms = _extract_symbols_regex(TS_CODE, "typescript")
    s = _find(syms, "IUserRepo", "interface")
    assert s is not None


def test_ts_regex_type_alias():
    syms = _extract_symbols_regex(TS_CODE, "typescript")
    s = _find(syms, "UserId", "type")
    assert s is not None


def test_ts_regex_enum():
    syms = _extract_symbols_regex(TS_CODE, "typescript")
    s = _find(syms, "Role", "enum")
    assert s is not None


def test_ts_regex_class():
    syms = _extract_symbols_regex(TS_CODE, "typescript")
    s = _find(syms, "UserRepo", "class")
    assert s is not None


# ---------------------------------------------------------------------------
# C# — regex fallback
# ---------------------------------------------------------------------------

CS_CODE = """\
namespace BpoPilot.Services
{
    public class OrderService
    {
        public void ProcessOrder(int id)
        {
        }
    }

    public interface IOrderService
    {
        void ProcessOrder(int id);
    }

    public enum OrderStatus
    {
        Pending,
        Completed,
    }
}
"""


def test_cs_regex_class():
    syms = _extract_symbols_regex(CS_CODE, "csharp")
    s = _find(syms, "OrderService", "class")
    assert s is not None


def test_cs_regex_interface():
    syms = _extract_symbols_regex(CS_CODE, "csharp")
    s = _find(syms, "IOrderService", "interface")
    assert s is not None


def test_cs_regex_enum():
    syms = _extract_symbols_regex(CS_CODE, "csharp")
    s = _find(syms, "OrderStatus", "enum")
    assert s is not None


def test_cs_regex_method():
    syms = _extract_symbols_regex(CS_CODE, "csharp")
    s = _find(syms, "ProcessOrder", "method")
    assert s is not None


def test_cs_namespace_captured():
    syms = _extract_symbols_regex(CS_CODE, "csharp")
    class_sym = _find(syms, "OrderService", "class")
    assert class_sym is not None
    assert class_sym.namespace == "BpoPilot.Services"


# ---------------------------------------------------------------------------
# Empty / unsupported language
# ---------------------------------------------------------------------------

def test_empty_code_returns_no_symbols():
    assert _extract_symbols_regex("", "python") == []


def test_unsupported_language_returns_empty():
    syms = extract_symbols("x = 1", "ruby")
    assert syms == []


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

DUPE_CODE = """\
class Foo:
    pass

class Foo:
    pass
"""


def test_deduplication():
    syms = _extract_symbols_regex(DUPE_CODE, "python")
    foo_syms = [s for s in syms if s.name == "Foo"]
    # Both lines are different line_start so both should appear,
    # but neither should be duplicated at the same (name, kind, line).
    assert len(foo_syms) == len({(s.name, s.kind, s.line_start) for s in foo_syms})


# ---------------------------------------------------------------------------
# tree-sitter (skipped if not installed)
# ---------------------------------------------------------------------------

_ts_available = importlib.util.find_spec("tree_sitter") is not None
_ts_python_available = importlib.util.find_spec("tree_sitter_python") is not None


@pytest.mark.skipif(
    not (_ts_available and _ts_python_available),
    reason="tree-sitter or tree-sitter-python not installed",
)
def test_treesitter_python_line_numbers():
    from rag_indexer.symbol_extractor import _extract_symbols_treesitter
    syms = _extract_symbols_treesitter(PYTHON_CODE, "python")
    s = _find(syms, "MyService", "class")
    assert s is not None
    assert s.line_start == 1
    assert s.line_end is not None
    assert s.line_end >= s.line_start


@pytest.mark.skipif(
    not (_ts_available and _ts_python_available),
    reason="tree-sitter or tree-sitter-python not installed",
)
def test_treesitter_python_signature():
    from rag_indexer.symbol_extractor import _extract_symbols_treesitter
    syms = _extract_symbols_treesitter(PYTHON_CODE, "python")
    s = _find(syms, "MyService", "class")
    assert s is not None
    assert s.signature is not None
    assert "MyService" in s.signature


# ---------------------------------------------------------------------------
# Fallback on ImportError
# ---------------------------------------------------------------------------

def test_fallback_on_import_error(monkeypatch):
    """If tree-sitter cannot be imported, extract_symbols falls back to regex."""
    import rag_indexer.symbol_extractor as mod

    original = mod._extract_symbols_treesitter

    def _raise(*args, **kwargs):
        raise ImportError("simulated tree-sitter unavailable")

    monkeypatch.setattr(mod, "_extract_symbols_treesitter", _raise)

    syms = extract_symbols(PYTHON_CODE, "python")
    # Should still return results from regex fallback
    assert any(s.name == "MyService" for s in syms)

    monkeypatch.setattr(mod, "_extract_symbols_treesitter", original)
