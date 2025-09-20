"""Lightweight YAML loader used when PyYAML is unavailable.

This module intentionally implements only the subset of YAML syntax used by
the project configuration files.  It supports:

* dictionaries and lists expressed with indentation
* scalar values (strings, numbers, booleans, and ``null``)
* inline list/dict literals (which are compatible with :func:`ast.literal_eval`)
* comments beginning with ``#`` outside of quoted strings

When PyYAML is installed the application will continue to use it.  The
classes and functions here exist purely as a fallback so that the application
can still validate configuration files in minimal environments (such as the
automated execution sandbox used for the kata) without requiring external
dependencies.
"""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

__all__ = ["safe_load", "dump", "YAMLError"]


class YAMLError(Exception):
    """Raised when the lightweight YAML parser encounters invalid input."""


def safe_load(stream: Any) -> Any:
    """Parse YAML content into Python data structures.

    Args:
        stream: Either a text string or a file-like object with ``read``.

    Returns:
        The parsed Python data structure (``dict``/``list``/scalar).
    """

    parser = _SimpleYAMLParser()
    try:
        return parser.load(stream)
    except Exception as exc:  # pragma: no cover - defensive guard
        raise YAMLError(str(exc)) from exc


def dump(data: Any, stream: Any = None, *, indent: int = 2, **_: Any) -> str:
    """Serialise data to a YAML string.

    The fallback implementation emits JSON-formatted output which is valid YAML
    1.2.  This keeps the implementation compact while remaining compatible with
    the rest of the project.
    """

    text = json.dumps(data, indent=indent)
    if stream is not None:
        stream.write(text)
        return text
    return text


@dataclass
class _Line:
    indent: int
    content: str


class _SimpleYAMLParser:
    """Very small indentation-based YAML parser."""

    def load(self, stream: Any) -> Any:
        if hasattr(stream, "read"):
            text = stream.read()
        else:
            text = str(stream)

        stripped = text.lstrip()
        if not stripped:
            return None
        if stripped[0] in "[{":
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                pass

        self._lines = self._prepare_lines(text)
        self._index = 0

        if not self._lines:
            return None

        return self._parse_block(self._lines[0].indent)

    def _prepare_lines(self, text: str) -> List[_Line]:
        lines: List[_Line] = []
        for raw_line in text.splitlines():
            line = self._strip_comment(raw_line.rstrip("\n"))
            if not line.strip():
                continue
            indent = len(line) - len(line.lstrip(" "))
            content = line.strip()
            lines.append(_Line(indent=indent, content=content))
        return lines

    def _strip_comment(self, line: str) -> str:
        in_single = False
        in_double = False
        for idx, char in enumerate(line):
            if char == "'" and not in_double and (idx == 0 or line[idx - 1] != "\\"):
                in_single = not in_single
            elif char == '"' and not in_single and (idx == 0 or line[idx - 1] != "\\"):
                in_double = not in_double
            elif char == "#" and not in_single and not in_double:
                return line[:idx].rstrip()
        return line.rstrip()

    def _parse_block(self, current_indent: int) -> Any:
        container_type: Optional[str] = None
        mapping: dict[str, Any] = {}
        sequence: List[Any] = []

        while self._index < len(self._lines):
            line = self._lines[self._index]
            if line.indent < current_indent:
                break
            if line.indent > current_indent:
                raise YAMLError(f"Unexpected indentation at line {self._index + 1}")

            if line.content.startswith("- "):
                if container_type not in (None, "list"):
                    raise YAMLError("Cannot mix mappings and lists at the same level")
                container_type = "list"
                self._index += 1
                item_value = line.content[2:].strip()
                sequence.append(self._parse_sequence_item(item_value, current_indent))
            else:
                if container_type not in (None, "dict"):
                    raise YAMLError("Cannot mix lists and mappings at the same level")
                container_type = "dict"
                key, value_str = self._split_key_value(line.content)
                self._index += 1
                mapping[key] = self._parse_mapping_value(value_str, current_indent)

        if container_type == "list":
            return sequence
        return mapping

    def _parse_sequence_item(self, value_str: str, parent_indent: int) -> Any:
        if not value_str:
            return self._parse_nested(parent_indent)

        if value_str.startswith(("'", '"', "[", "{")):
            return self._parse_scalar(value_str)

        if ":" in value_str:
            key, remainder = self._split_key_value(value_str)
            mapping: dict[str, Any] = {}
            mapping[key] = self._parse_inline_value(remainder, parent_indent)

            if self._index < len(self._lines):
                next_line = self._lines[self._index]
                if next_line.indent > parent_indent:
                    nested = self._parse_block(next_line.indent)
                    if not isinstance(nested, dict):
                        raise YAMLError("Expected mapping content for list item")
                    mapping.update(nested)
            return mapping

        return self._parse_scalar(value_str)

    def _parse_mapping_value(self, value_str: str, parent_indent: int) -> Any:
        if not value_str:
            return self._parse_nested(parent_indent)
        return self._parse_scalar(value_str)

    def _parse_inline_value(self, value_str: str, parent_indent: int) -> Any:
        if value_str:
            return self._parse_scalar(value_str)
        return self._parse_nested(parent_indent)

    def _parse_nested(self, parent_indent: int) -> Any:
        if self._index >= len(self._lines):
            return None
        next_line = self._lines[self._index]
        if next_line.indent <= parent_indent:
            return None
        return self._parse_block(next_line.indent)

    def _parse_scalar(self, value_str: str) -> Any:
        value_str = value_str.strip()
        if not value_str:
            return None

        lowered = value_str.lower()
        if lowered in {"null", "none", "~"}:
            return None
        if lowered in {"true", "yes", "on"}:
            return True
        if lowered in {"false", "no", "off"}:
            return False

        if value_str.startswith(("'", '"', "[", "{", "(")):
            try:
                return ast.literal_eval(value_str)
            except (ValueError, SyntaxError) as exc:
                raise YAMLError(f"Invalid literal: {value_str}") from exc

        # Numbers (ints first, then floats)
        try:
            if value_str.startswith(("-", "+")) and value_str[1:].isdigit():
                return int(value_str)
            if value_str.isdigit():
                return int(value_str)
            return float(value_str)
        except ValueError:
            pass

        return value_str

    def _split_key_value(self, content: str) -> Tuple[str, str]:
        in_single = False
        in_double = False
        for idx, char in enumerate(content):
            if char == "'" and not in_double and (idx == 0 or content[idx - 1] != "\\"):
                in_single = not in_single
            elif char == '"' and not in_single and (idx == 0 or content[idx - 1] != "\\"):
                in_double = not in_double
            elif char == ":" and not in_single and not in_double:
                key = content[:idx].strip()
                value = content[idx + 1 :].strip()
                if not key:
                    raise YAMLError(f"Empty key in mapping entry: {content}")
                return key, value
        raise YAMLError(f"Expected ':' in mapping entry: {content}")

