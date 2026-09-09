from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SourceChunk:
    text: str
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass
class Index:
    chunks: list[SourceChunk]
    collection: object
    chunk_by_id: dict[str, SourceChunk]
    partial: bool = False
    source_paths: list[str] = field(default_factory=list)
    collection_name: str = ""
