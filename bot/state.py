from __future__ import annotations

import json
import os
import uuid
from datetime import datetime
from typing import Any, Dict, List, Sequence, Optional

# Silence missing stubs for SQLAlchemy in type checkers
from sqlalchemy import (  # type: ignore
    Column,
    DateTime,
    String,
    Text,
    create_engine,
    delete,
    select,
    text,
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session  # type: ignore

Base = declarative_base()

# Default character limit for conversation history (roughly ~10k tokens)
_DEFAULT_MAX_HISTORY_CHARS = 40_000


class _Message(Base):  # noqa: D101 (internal class)
    __tablename__ = "history"

    id: str = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))  # type: ignore[assignment]
    channel_id: str = Column(String, nullable=False, index=True)  # type: ignore[assignment]
    created_at: datetime = Column(
        DateTime, nullable=False, server_default=text("CURRENT_TIMESTAMP")
    )  # type: ignore[assignment]
    message_json: str = Column(String, nullable=False)  # type: ignore[assignment]


class _Memory(Base):  # noqa: D101 (internal class)
    """Long-term memory storage for important information the agent should remember."""
    __tablename__ = "memories"

    id: str = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))  # type: ignore[assignment]
    channel_id: str = Column(String, nullable=False, index=True)  # type: ignore[assignment]
    content: str = Column(Text, nullable=False)  # type: ignore[assignment]
    created_at: datetime = Column(
        DateTime, nullable=False, server_default=text("CURRENT_TIMESTAMP")
    )  # type: ignore[assignment]
    updated_at: datetime = Column(
        DateTime, nullable=False, server_default=text("CURRENT_TIMESTAMP")
    )  # type: ignore[assignment]


class StateStore:
    """Lightweight persistence layer for the Agent's conversation history and long-term memories."""

    def __init__(self, maximum_history_chars: int | None = None):
        self._maximum_history_chars = maximum_history_chars or _DEFAULT_MAX_HISTORY_CHARS
        db_path = os.path.join(os.path.dirname(__file__), "..", "agent_history.db")
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._engine = create_engine(f"sqlite:///{db_path}", future=True, echo=False)
        Base.metadata.create_all(self._engine)

        self._Session: sessionmaker[Session] = sessionmaker(
            self._engine, expire_on_commit=False, class_=Session
        )

    # ------------------------------------------------------------------
    # Conversation History API
    # ------------------------------------------------------------------
    def load_history(self, channel_id: str) -> List[Dict[str, Any]]:
        """Load messages for a specific channel in chronological order."""
        history: List[Dict[str, Any]] = []
        with self._Session() as session:
            stmt = (
                select(_Message)
                .where(_Message.channel_id == channel_id)  # type: ignore[arg-type]
                .order_by(_Message.created_at)  # type: ignore[arg-type]
            )
            rows = session.scalars(stmt).all()
            for row in rows:
                try:
                    history.append(json.loads(row.message_json))
                except json.JSONDecodeError:
                    continue
        return history

    def append(
        self, channel_id: str, message: Dict[str, Any], auto_trim: bool = True
    ) -> List[Dict[str, Any]]:
        """Persist a single message for a channel."""
        message_json = json.dumps(message)
        with self._Session() as session:
            session.add(
                _Message(
                    channel_id=channel_id,
                    message_json=message_json,
                )
            )
            session.commit()

            if auto_trim:
                return self.trim_history(channel_id)
            else:
                return self.load_history(channel_id)

    def reset(self, channel_id: str) -> List[Dict[str, Any]]:
        """Delete stored messages.

        Args:
            channel_id (str | None): If provided, only messages for that channel
                are deleted. If ``None`` delete all conversation data.
        """
        with self._Session() as session:
            if channel_id is None:
                session.execute(delete(_Message))
            else:
                session.execute(
                    delete(_Message).where(_Message.channel_id == channel_id)  # type: ignore[arg-type]
                )
            session.commit()

            return self.load_history(channel_id)

    def trim_history(self, channel_id: str) -> List[Dict[str, Any]]:
        """Trim old messages to keep history within the character limit.

        This removes the oldest messages while preserving the most recent
        conversations up to the character limit. The trimming respects
        conversation boundaries by stopping at user messages.

        Args:
            channel_id (str): The channel to trim messages for.

        Returns:
            Updated message history after trimming.
        """
        if self._maximum_history_chars is None:
            return self.load_history(channel_id)

        with self._Session() as session:
            # Get all messages for the channel in chronological order
            stmt = (
                select(_Message)
                .where(_Message.channel_id == channel_id)  # type: ignore[arg-type]
                .order_by(_Message.created_at)  # type: ignore[arg-type]
            )
            rows = session.scalars(stmt).all()

            # Get the message IDs to keep based on character limit
            messages_to_keep = self._get_message_ids_by_char_limit(
                rows, self._maximum_history_chars
            )

            # Delete messages that are not in the keep list
            if messages_to_keep:
                session.execute(
                    delete(_Message).where(
                        (_Message.channel_id == channel_id)  # type: ignore[arg-type]
                        & (~_Message.id.in_(messages_to_keep))  # type: ignore[arg-type]
                    )
                )
            else:
                # If no messages to keep, delete all for this channel
                session.execute(
                    delete(_Message).where(_Message.channel_id == channel_id)  # type: ignore[arg-type]
                )

            session.commit()

        # Return the updated history after trimming
        return self.load_history(channel_id)

    def _get_message_ids_by_char_limit(
        self, rows: Sequence[_Message], max_chars: int
    ) -> List[str]:
        """Get the message IDs to keep based on the character limit.

        Args:
            rows: List of message rows in chronological order
            max_chars: Maximum total characters to keep

        Returns:
            List of message IDs to keep
        """
        total_chars = 0
        messages_to_keep: List[str] = []
        last_user_message_idx: Optional[int] = None

        # Go through messages in reverse order (newest first) to keep recent ones
        reversed_rows = list(reversed(rows))
        for idx, row in enumerate(reversed_rows):
            char_count = len(row.message_json)
            
            # Check if adding this message would exceed the limit
            if total_chars + char_count > max_chars:
                # If we're at a user message boundary, we can stop here
                # Otherwise, continue to find a good stopping point
                if last_user_message_idx is not None:
                    # Trim to the last complete user message exchange
                    messages_to_keep = messages_to_keep[:last_user_message_idx + 1]
                break
            
            total_chars += char_count
            messages_to_keep.append(row.id)
            
            # Track user message boundaries for clean trimming
            try:
                message = json.loads(row.message_json)
                if message.get("role") == "user":
                    last_user_message_idx = idx
            except json.JSONDecodeError:
                continue

        return messages_to_keep

    # ------------------------------------------------------------------
    # Long-Term Memory API
    # ------------------------------------------------------------------
    def load_memories(self, channel_id: str) -> List[Dict[str, Any]]:
        """Load all long-term memories for a specific channel.

        Args:
            channel_id: The channel to load memories for.

        Returns:
            List of memory dictionaries with id, content, created_at, updated_at.
        """
        memories: List[Dict[str, Any]] = []
        with self._Session() as session:
            stmt = (
                select(_Memory)
                .where(_Memory.channel_id == channel_id)  # type: ignore[arg-type]
                .order_by(_Memory.created_at)  # type: ignore[arg-type]
            )
            rows = session.scalars(stmt).all()
            for row in rows:
                memories.append({
                    "id": row.id,
                    "content": row.content,
                    "created_at": row.created_at.isoformat() if row.created_at else None,
                    "updated_at": row.updated_at.isoformat() if row.updated_at else None,
                })
        return memories

    def add_memory(self, channel_id: str, content: str) -> Dict[str, Any]:
        """Add a new long-term memory for a channel.

        Args:
            channel_id: The channel to add the memory to.
            content: The memory content to store.

        Returns:
            The created memory record.
        """
        with self._Session() as session:
            memory = _Memory(channel_id=channel_id, content=content)
            session.add(memory)
            session.commit()
            session.refresh(memory)
            return {
                "id": memory.id,
                "content": memory.content,
                "created_at": memory.created_at.isoformat() if memory.created_at else None,
                "updated_at": memory.updated_at.isoformat() if memory.updated_at else None,
            }

    def update_memory(self, memory_id: str, content: str) -> Optional[Dict[str, Any]]:
        """Update an existing long-term memory.

        Args:
            memory_id: The ID of the memory to update.
            content: The new content for the memory.

        Returns:
            The updated memory record, or None if not found.
        """
        with self._Session() as session:
            stmt = select(_Memory).where(_Memory.id == memory_id)  # type: ignore[arg-type]
            memory = session.scalars(stmt).first()
            if memory is None:
                return None
            
            memory.content = content
            memory.updated_at = datetime.utcnow()
            session.commit()
            session.refresh(memory)
            return {
                "id": memory.id,
                "content": memory.content,
                "created_at": memory.created_at.isoformat() if memory.created_at else None,
                "updated_at": memory.updated_at.isoformat() if memory.updated_at else None,
            }

    def delete_memory(self, memory_id: str) -> bool:
        """Delete a long-term memory by ID.

        Args:
            memory_id: The ID of the memory to delete.

        Returns:
            True if the memory was deleted, False if not found.
        """
        with self._Session() as session:
            stmt = select(_Memory).where(_Memory.id == memory_id)  # type: ignore[arg-type]
            memory = session.scalars(stmt).first()
            if memory is None:
                return False
            
            session.delete(memory)
            session.commit()
            return True

    def clear_memories(self, channel_id: str) -> None:
        """Delete all long-term memories for a channel.

        Args:
            channel_id: The channel to clear memories for.
        """
        with self._Session() as session:
            session.execute(
                delete(_Memory).where(_Memory.channel_id == channel_id)  # type: ignore[arg-type]
            )
            session.commit()

    def get_memories_text(self, channel_id: str) -> str:
        """Get formatted text of all memories for inclusion in instructions.

        Args:
            channel_id: The channel to get memories for.

        Returns:
            Formatted string of memories, or empty string if none exist.
        """
        memories = self.load_memories(channel_id)
        if not memories:
            return ""
        
        memory_lines = []
        for i, memory in enumerate(memories, 1):
            memory_lines.append(f"{i}. [{memory['id'][:8]}] {memory['content']}")
        
        return "\n".join(memory_lines)
