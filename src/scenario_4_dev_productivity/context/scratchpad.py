"""Scratchpad file — cross-context persistence for agent findings.

The spec calls out two failure modes the scratchpad fixes:

* **Context compression.** The agent's window gets compressed by
  :class:`ContextCompactor` and verbose history is lost. Anything the
  agent wrote to the scratchpad survives — it's the only durable
  surface for "key findings" across a /compact.
* **Crash recovery.** A coordinator that has been running for hours
  can dump its state to the scratchpad, die, and resume from the
  scratchpad instead of starting over.

The class is deliberately small: read, write, append, clear. The
file format is plain Markdown so a human can read it without a tool.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ScratchpadEntry(BaseModel):
    """One entry written to the scratchpad.

    The agent attaches a short ``topic`` and a ``body``. Entries are
    appended in insertion order; consumers can filter by topic.
    """

    model_config = ConfigDict(extra="forbid")

    topic: str = Field(min_length=1, description="Short topic heading for the entry")
    body: str = Field(min_length=1, description="The entry's Markdown body")
    source: str | None = Field(
        default=None,
        description="Optional origin tag (agent name, tool name) for debugging",
    )


class ScratchpadManager:
    """Read/write/append a Markdown scratchpad file.

    The manager holds a path; the path is created lazily on first
    write. All operations are atomic per call (single ``open`` + write)
    — concurrent writers can race, but a single agent doesn't.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    # -- properties -------------------------------------------------------

    @property
    def path(self) -> Path:
        """The scratchpad file path. The file may not exist yet."""
        return self._path

    @property
    def exists(self) -> bool:
        """True when the scratchpad file is on disk."""
        return self._path.exists()

    # -- reads ------------------------------------------------------------

    def read(self) -> str:
        """Return the full scratchpad contents, or ``""`` if it doesn't exist."""
        if not self._path.exists():
            return ""
        return self._path.read_text(encoding="utf-8")

    def read_entries(self) -> list[ScratchpadEntry]:
        """Parse the scratchpad back into entries.

        The expected format is::

            ## <topic>
            <body>
            <!-- source: <source> -->

        Unparseable blocks are skipped — the scratchpad is best-effort
        state, not a database. If you need durable state, use a real
        database.
        """
        text = self.read()
        if not text.strip():
            return []
        return list(_parse_entries(text))

    # -- writes -----------------------------------------------------------

    def write(self, content: str) -> None:
        """Replace the scratchpad with ``content``.

        Creates the parent directory if it doesn't exist.
        """
        self._ensure_parent()
        self._path.write_text(content, encoding="utf-8")

    def clear(self) -> None:
        """Delete the scratchpad file. No-op when it doesn't exist."""
        if self._path.exists():
            self._path.unlink()

    def append(self, entry: ScratchpadEntry) -> None:
        """Append a single entry to the scratchpad.

        Each entry is rendered as a level-2 Markdown section, so
        ``read_entries`` can recover them.
        """
        self._ensure_parent()
        block = _render_entry(entry)
        with self._path.open("a", encoding="utf-8") as fh:
            if self._path.exists() and self._path.stat().st_size > 0:
                fh.write("\n")
            fh.write(block)

    def append_finding(self, topic: str, body: str, source: str | None = None) -> ScratchpadEntry:
        """Convenience: build an entry and append it. Returns the entry written."""
        entry = ScratchpadEntry(topic=topic, body=body, source=source)
        self.append(entry)
        return entry

    # -- internals --------------------------------------------------------

    def _ensure_parent(self) -> None:
        parent = self._path.parent
        if str(parent) and not parent.exists():
            parent.mkdir(parents=True, exist_ok=True)


# -- module-level helpers (used by tests) ---------------------------------


def _render_entry(entry: ScratchpadEntry) -> str:
    """Render a :class:`ScratchpadEntry` as a Markdown block."""
    body = entry.body.rstrip()
    if entry.source:
        return f"## {entry.topic}\n{body}\n<!-- source: {entry.source} -->\n"
    return f"## {entry.topic}\n{body}\n"


def _parse_entries(text: str) -> Any:
    """Yield :class:`ScratchpadEntry` from a Markdown document.

    Split on ``## `` headings. Each section is the topic heading
    followed by lines until the next ``## `` or end-of-file. The
    optional ``<!-- source: ... -->`` footer is peeled off.
    """
    sections: list[ScratchpadEntry] = []
    current_topic: str | None = None
    current_lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if line.startswith("## "):
            if current_topic is not None:
                sections.append(_build_entry(current_topic, current_lines))
            current_topic = line[3:].strip()
            current_lines = []
        else:
            current_lines.append(raw_line)
    if current_topic is not None:
        sections.append(_build_entry(current_topic, current_lines))
    return sections


def _build_entry(topic: str, lines: list[str]) -> ScratchpadEntry:
    body = "\n".join(lines).strip()
    source: str | None = None
    if body.endswith("-->"):
        marker = "<!-- source:"
        idx = body.rfind(marker)
        if idx != -1:
            tail = body[idx + len(marker) : -3].strip()
            source = tail or None
            body = body[:idx].rstrip()
    if not body:
        # Drop empty entries — they're noise.
        return ScratchpadEntry(topic=topic, body="(empty)", source=source)
    return ScratchpadEntry(topic=topic, body=body, source=source)


__all__ = ["ScratchpadManager", "ScratchpadEntry"]
