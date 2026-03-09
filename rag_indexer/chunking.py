from __future__ import annotations

from dataclasses import dataclass
import bisect
import re


@dataclass
class Chunk:
    text: str
    char_start: int
    char_end: int
    line_start: int
    line_end: int
    section_path: str = ""
    is_header: bool = False


def chunk_markdown(
    text: str,
    max_chars: int,
    overlap: int,
    min_chunk_chars: int,
) -> list[Chunk]:
    line_index = _build_line_index(text)
    lines = text.splitlines(keepends=True)
    sections: list[tuple[str, int, str]] = []
    current: list[str] = []
    current_start = 0
    heading_re = re.compile(r"^\s*(?P<hashes>#{1,6})\s+(?P<title>.+)")
    heading_stack: list[tuple[int, str]] = []
    current_section_path = ""
    cursor = 0
    for line in lines:
        match = heading_re.match(line)
        if match:
            if current:
                sections.append(("".join(current), current_start, current_section_path))
            level = len(match.group("hashes"))
            title = match.group("title").strip()
            heading_stack = [item for item in heading_stack if item[0] < level]
            heading_stack.append((level, title))
            current_section_path = " > ".join(item[1] for item in heading_stack)
            current = [line]
            current_start = cursor
        else:
            if not current:
                current_start = cursor
            current.append(line)
        cursor += len(line)
    if current:
        sections.append(("".join(current), current_start, current_section_path))
    chunks: list[Chunk] = []
    for section_text, section_start, section_path in sections:
        if len(section_text) <= max_chars:
            chunks.append(
                _make_chunk(
                    section_text,
                    section_start,
                    section_start + len(section_text),
                    line_index,
                    section_path=section_path,
                )
            )
        else:
            chunks.extend(
                chunk_text_with_offsets(
                    section_text,
                    section_start,
                    line_index,
                    max_chars,
                    overlap,
                    min_chunk_chars,
                    section_path=section_path,
                )
            )
    return _filter_chunks(chunks, min_chunk_chars)


def chunk_code(
    text: str,
    max_chars: int,
    min_chunk_chars: int,
) -> list[Chunk]:
    boundary_re = re.compile(
        r"^\s*(?:"
        r"(?:def |class |function |export function|export class)"
        r"|(?:(?:public|private|protected|internal|sealed|abstract|static|partial)\s+)*"
        r"(?:class|interface|struct|enum)\s+\w+"
        r"|(?:public|private|protected|internal)\s+"
        r"(?:static\s+)?(?:async\s+)?[\w<>\[\],\s]+\s+\w+\s*\()",
    )
    line_index = _build_line_index(text)
    lines = text.splitlines(keepends=True)
    chunks: list[Chunk] = []
    current: list[str] = []
    current_len = 0
    current_start = 0
    cursor = 0
    for line in lines:
        if boundary_re.match(line) and current_len >= min_chunk_chars:
            chunk_text = "".join(current)
            if chunk_text.strip():
                chunks.append(
                    _make_chunk(
                        chunk_text,
                        current_start,
                        current_start + len(chunk_text),
                        line_index,
                    )
                )
            current = [line]
            current_len = len(line)
            current_start = cursor
            cursor += len(line)
            continue
        if not current:
            current_start = cursor
        current.append(line)
        current_len += len(line)
        cursor += len(line)
        if current_len >= max_chars:
            chunk_text = "".join(current)
            if chunk_text.strip():
                chunks.append(
                    _make_chunk(
                        chunk_text,
                        current_start,
                        current_start + len(chunk_text),
                        line_index,
                    )
                )
            current = []
            current_len = 0
    if current:
        chunk_text = "".join(current)
        if chunk_text.strip():
            chunks.append(
                _make_chunk(
                    chunk_text,
                    current_start,
                    current_start + len(chunk_text),
                    line_index,
                )
            )
    return _filter_chunks(chunks, min_chunk_chars)


def chunk_text(
    text: str,
    max_chars: int,
    overlap: int,
    min_chunk_chars: int,
) -> list[Chunk]:
    line_index = _build_line_index(text)
    return chunk_text_with_offsets(
        text,
        0,
        line_index,
        max_chars,
        overlap,
        min_chunk_chars,
        section_path="",
    )


def chunk_text_with_offsets(
    text: str,
    base_offset: int,
    line_index: list[int],
    max_chars: int,
    overlap: int,
    min_chunk_chars: int,
    section_path: str = "",
) -> list[Chunk]:
    if max_chars <= 0:
        chunk = _make_chunk(
            text,
            base_offset,
            base_offset + len(text),
            line_index,
            section_path=section_path,
        )
        return [chunk] if chunk.text.strip() else []
    overlap = max(0, min(overlap, max_chars - 1))
    chunks: list[Chunk] = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(text_len, start + max_chars)
        chunk_text = text[start:end]
        if chunk_text.strip():
            chunks.append(
                _make_chunk(
                    chunk_text,
                    base_offset + start,
                    base_offset + end,
                    line_index,
                    section_path=section_path,
                )
            )
        if end >= text_len:
            break
        start = max(0, end - overlap)
    return _filter_chunks(chunks, min_chunk_chars)


def _filter_chunks(chunks: list[Chunk], min_chunk_chars: int) -> list[Chunk]:
    cleaned = [chunk for chunk in chunks if chunk.text.strip()]
    if not cleaned:
        return []
    if min_chunk_chars <= 0:
        return cleaned
    filtered = [chunk for chunk in cleaned if len(chunk.text.strip()) >= min_chunk_chars]
    return filtered or cleaned


def _build_line_index(text: str) -> list[int]:
    if not text:
        return [0]
    starts = [0]
    for index, char in enumerate(text):
        if char == "\n":
            starts.append(index + 1)
    return starts


def _line_no_at(line_index: list[int], char_pos: int) -> int:
    if char_pos < 0:
        return 1
    if not line_index:
        return 1
    return bisect.bisect_right(line_index, char_pos)


def _make_chunk(
    text: str,
    char_start: int,
    char_end: int,
    line_index: list[int],
    section_path: str = "",
    is_header: bool = False,
) -> Chunk:
    safe_end = max(char_start, char_end - 1)
    line_start = _line_no_at(line_index, char_start)
    line_end = _line_no_at(line_index, safe_end)
    return Chunk(
        text=text.strip("\n"),
        char_start=char_start,
        char_end=char_end,
        line_start=line_start,
        line_end=line_end,
        section_path=section_path,
        is_header=is_header,
    )
