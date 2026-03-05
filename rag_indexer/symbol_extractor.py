"""
Symbol extraction for code files.

Primary: tree-sitter AST parsing (precise line numbers, signatures, scope).
Fallback: regex-based extraction (same regexes used in ingest.py header building).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class SymbolInfo:
    name: str
    kind: str  # class | function | method | interface | enum | struct | type
    namespace: Optional[str]
    line_start: int  # 1-based
    line_end: Optional[int]
    signature: Optional[str]


# ---------------------------------------------------------------------------
# Regex fallback (mirrors patterns in ingest.py)
# ---------------------------------------------------------------------------

_CS_SYMBOL_RE = re.compile(
    r"^\s*(?:(?:public|private|protected|internal|sealed|abstract|static|partial)\s+)*"
    r"(class|interface|struct|enum)\s+(?P<name>[A-Za-z_]\w*)",
    re.MULTILINE,
)
_CS_NAMESPACE_RE = re.compile(r"^\s*namespace\s+(?P<name>[A-Za-z_][\w\.]*)", re.MULTILINE)
_CS_METHOD_RE = re.compile(
    r"^\s*(?:public|private|protected|internal)\s+"
    r"(?:static\s+)?(?:async\s+)?[\w<>\[\],\s]+\s+(?P<name>[A-Za-z_]\w*)\s*\(",
    re.MULTILINE,
)
_TS_SYMBOL_RE = re.compile(
    r"^\s*(?:export\s+)?(class|interface|enum|type)\s+(?P<name>[A-Za-z_]\w*)",
    re.MULTILINE,
)
_PY_CLASS_RE = re.compile(
    r"^(?P<indent>[^\S\n]*)class\s+(?P<name>[A-Za-z_]\w*)\s*(?:\(|:)",
    re.MULTILINE,
)
_PY_FUNC_RE = re.compile(
    r"^(?P<indent>[^\S\n]*)(?:async\s+)?def\s+(?P<name>[A-Za-z_]\w*)\s*\(",
    re.MULTILINE,
)
_JS_CLASS_RE = re.compile(
    r"^\s*(?:export\s+)?class\s+(?P<name>[A-Za-z_$][\w$]*)",
    re.MULTILINE,
)
_JS_FUNCTION_RE = re.compile(
    r"^\s*(?:export\s+)?(?:async\s+)?function\s+(?P<name>[A-Za-z_$][\w$]*)\s*\(",
    re.MULTILINE,
)
_JS_ASSIGNED_FUNCTION_RE = re.compile(
    r"^\s*(?:export\s+)?(?:const|let|var)\s+(?P<name>[A-Za-z_$][\w$]*)\s*=\s*"
    r"(?:async\s+)?(?:function\b|\([^)]*\)\s*=>|[A-Za-z_$][\w$]*\s*=>)",
    re.MULTILINE,
)


def _line_of_match(text: str, match: re.Match) -> int:
    """Return 1-based line number for a regex match start."""
    return text[: match.start()].count("\n") + 1


def _extract_symbols_regex(text: str, language: str) -> list[SymbolInfo]:
    """Regex-based fallback extraction."""
    symbols: list[SymbolInfo] = []
    lang = language.strip().lower() if language else ""

    if lang in {"", "csharp", "aspnet"}:
        ns_match = _CS_NAMESPACE_RE.search(text)
        namespace = ns_match.group("name") if ns_match else None
        for m in _CS_SYMBOL_RE.finditer(text):
            kind = m.group(1).lower()
            symbols.append(SymbolInfo(
                name=m.group("name"),
                kind=kind,
                namespace=namespace,
                line_start=_line_of_match(text, m),
                line_end=None,
                signature=None,
            ))
        for m in _CS_METHOD_RE.finditer(text):
            symbols.append(SymbolInfo(
                name=m.group("name"),
                kind="method",
                namespace=namespace,
                line_start=_line_of_match(text, m),
                line_end=None,
                signature=None,
            ))

    if lang in {"", "typescript"}:
        for m in _TS_SYMBOL_RE.finditer(text):
            kind = m.group(1).lower()
            symbols.append(SymbolInfo(
                name=m.group("name"),
                kind=kind,
                namespace=None,
                line_start=_line_of_match(text, m),
                line_end=None,
                signature=None,
            ))

    if lang in {"", "python"}:
        for m in _PY_CLASS_RE.finditer(text):
            symbols.append(SymbolInfo(
                name=m.group("name"),
                kind="class",
                namespace=None,
                line_start=_line_of_match(text, m),
                line_end=None,
                signature=None,
            ))
        for m in _PY_FUNC_RE.finditer(text):
            indent = m.group("indent")
            kind = "method" if indent else "function"
            symbols.append(SymbolInfo(
                name=m.group("name"),
                kind=kind,
                namespace=None,
                line_start=_line_of_match(text, m),
                line_end=None,
                signature=None,
            ))

    if lang in {"", "javascript", "typescript"}:
        for m in _JS_CLASS_RE.finditer(text):
            symbols.append(SymbolInfo(
                name=m.group("name"),
                kind="class",
                namespace=None,
                line_start=_line_of_match(text, m),
                line_end=None,
                signature=None,
            ))
        for m in _JS_FUNCTION_RE.finditer(text):
            symbols.append(SymbolInfo(
                name=m.group("name"),
                kind="function",
                namespace=None,
                line_start=_line_of_match(text, m),
                line_end=None,
                signature=None,
            ))
        for m in _JS_ASSIGNED_FUNCTION_RE.finditer(text):
            symbols.append(SymbolInfo(
                name=m.group("name"),
                kind="function",
                namespace=None,
                line_start=_line_of_match(text, m),
                line_end=None,
                signature=None,
            ))

    # deduplicate by (name, kind, line_start)
    seen: set[tuple[str, str, int]] = set()
    result: list[SymbolInfo] = []
    for s in symbols:
        key = (s.name, s.kind, s.line_start)
        if key not in seen:
            seen.add(key)
            result.append(s)
    return result


# ---------------------------------------------------------------------------
# tree-sitter extraction
# ---------------------------------------------------------------------------

# Node type → kind mapping per language
_PY_NODE_KINDS: dict[str, str] = {
    "class_definition": "class",
    "function_definition": "function",
    "decorated_definition": "function",  # resolved below
}

_JS_NODE_KINDS: dict[str, str] = {
    "class_declaration": "class",
    "function_declaration": "function",
    "method_definition": "method",
    "interface_declaration": "interface",
    "type_alias_declaration": "type",
    "enum_declaration": "enum",
}

_TS_NODE_KINDS: dict[str, str] = {**_JS_NODE_KINDS}

_CS_NODE_KINDS: dict[str, str] = {
    "class_declaration": "class",
    "interface_declaration": "interface",
    "struct_declaration": "struct",
    "enum_declaration": "enum",
    "method_declaration": "method",
    "constructor_declaration": "method",
}


def _ts_language_for(language: str):
    """Import and return the tree-sitter Language object for the given lang string."""
    lang = language.strip().lower()
    if lang == "python":
        import tree_sitter_python as ts_lang  # type: ignore
        from tree_sitter import Language
        return Language(ts_lang.language())
    if lang == "javascript":
        import tree_sitter_javascript as ts_lang  # type: ignore
        from tree_sitter import Language
        return Language(ts_lang.language())
    if lang == "typescript":
        import tree_sitter_typescript as ts_lang  # type: ignore
        from tree_sitter import Language
        return Language(ts_lang.language_typescript())
    if lang in {"csharp", "aspnet"}:
        import tree_sitter_c_sharp as ts_lang  # type: ignore
        from tree_sitter import Language
        return Language(ts_lang.language())
    raise ImportError(f"No tree-sitter grammar for language: {language!r}")


def _node_kinds_for(language: str) -> dict[str, str]:
    lang = language.strip().lower()
    if lang == "python":
        return _PY_NODE_KINDS
    if lang == "javascript":
        return _JS_NODE_KINDS
    if lang == "typescript":
        return _TS_NODE_KINDS
    if lang in {"csharp", "aspnet"}:
        return _CS_NODE_KINDS
    return {}


def _find_child_field(node, field_name: str):
    """Return first child of node that matches field_name, or None."""
    try:
        return node.child_by_field_name(field_name)
    except Exception:
        return None


def _get_node_name(node, lines: list[str]) -> Optional[str]:
    """Extract identifier name from node via 'name' field."""
    name_node = _find_child_field(node, "name")
    if name_node is None:
        return None
    row = name_node.start_point[0]
    col_start = name_node.start_point[1]
    col_end = name_node.end_point[1]
    if row < len(lines):
        return lines[row][col_start:col_end]
    return None


def _get_signature(node, lines: list[str]) -> Optional[str]:
    """Extract the first line of a node as its signature."""
    row = node.start_point[0]
    if row < len(lines):
        return lines[row].strip()
    return None


def _extract_namespace_cs(root_node, lines: list[str]) -> Optional[str]:
    """Walk top-level to find namespace_declaration."""
    for child in root_node.children:
        if child.type == "namespace_declaration":
            name_node = _find_child_field(child, "name")
            if name_node:
                row = name_node.start_point[0]
                return lines[row][name_node.start_point[1]:name_node.end_point[1]] if row < len(lines) else None
    return None


def _walk_tree(node, target_types: set[str], results: list):
    """Recursively walk AST collecting nodes of target types."""
    if node.type in target_types:
        results.append(node)
    for child in node.children:
        _walk_tree(child, target_types, results)


def _extract_symbols_treesitter(text: str, language: str) -> list[SymbolInfo]:
    from tree_sitter import Parser  # type: ignore

    ts_lang = _ts_language_for(language)
    parser = Parser(ts_lang)
    tree = parser.parse(text.encode("utf-8", errors="replace"))
    lines = text.splitlines()
    node_kinds = _node_kinds_for(language)
    target_types = set(node_kinds.keys())

    lang_norm = language.strip().lower()
    namespace: Optional[str] = None
    if lang_norm in {"csharp", "aspnet"}:
        namespace = _extract_namespace_cs(tree.root_node, lines)

    collected_nodes: list = []
    _walk_tree(tree.root_node, target_types, collected_nodes)

    symbols: list[SymbolInfo] = []
    for node in collected_nodes:
        node_type = node.type
        # For decorated_definition in Python, look at the actual def child
        actual_node = node
        actual_type = node_type
        if lang_norm == "python" and node_type == "decorated_definition":
            for child in node.children:
                if child.type in {"function_definition", "class_definition"}:
                    actual_node = child
                    actual_type = child.type
                    break

        kind = node_kinds.get(actual_type, node_kinds.get(node_type, "function"))
        name = _get_node_name(actual_node, lines)
        if not name:
            continue

        # Distinguish method vs function for Python by indent
        if lang_norm == "python" and kind == "function":
            row = actual_node.start_point[0]
            if row < len(lines):
                indent = len(lines[row]) - len(lines[row].lstrip())
                if indent > 0:
                    kind = "method"

        signature = _get_signature(actual_node, lines)
        line_start = actual_node.start_point[0] + 1  # 1-based
        line_end = actual_node.end_point[0] + 1

        symbols.append(SymbolInfo(
            name=name,
            kind=kind,
            namespace=namespace,
            line_start=line_start,
            line_end=line_end,
            signature=signature,
        ))

    # deduplicate
    seen: set[tuple[str, str, int]] = set()
    result: list[SymbolInfo] = []
    for s in symbols:
        key = (s.name, s.kind, s.line_start)
        if key not in seen:
            seen.add(key)
            result.append(s)
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_symbols(text: str, language: str) -> list[SymbolInfo]:
    """
    Extract symbols from source code.

    Tries tree-sitter first for precise AST-based extraction.
    Falls back to regex if tree-sitter is unavailable or parse fails.
    """
    lang = (language or "").strip().lower()
    if lang not in {"python", "javascript", "typescript", "csharp", "aspnet"}:
        return _extract_symbols_regex(text, language)

    try:
        return _extract_symbols_treesitter(text, lang)
    except ImportError:
        return _extract_symbols_regex(text, language)
    except Exception:
        return _extract_symbols_regex(text, language)
