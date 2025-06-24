from __future__ import annotations

import json
import os
import uuid
from datetime import datetime
from typing import Any, Dict, List, Sequence

# Silence missing stubs for SQLAlchemy in type checkers
from sqlalchemy import (  # type: ignore
    Column,
    DateTime,
    String,
    create_engine,
    delete,
    select,
    text,
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session  # type: ignore

Base = declarative_base()


class _Message(Base):  # noqa: D101 (internal class)
    __tablename__ = "history"

    id: str = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))  # type: ignore[assignment]
    channel_id: str = Column(String, nullable=False, index=True)  # type: ignore[assignment]
    created_at: datetime = Column(
        DateTime, nullable=False, server_default=text("CURRENT_TIMESTAMP")
    )  # type: ignore[assignment]
    message_json: str = Column(String, nullable=False)  # type: ignore[assignment]


class StateStore:
    """Lightweight persistence layer for the Agent's conversation history."""

    def __init__(self, maximum_user_messages: int | None = None):
        self._maximum_user_messages = maximum_user_messages
        db_path = os.path.join(os.path.dirname(__file__), "..", "agent_history.db")
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._engine = create_engine(f"sqlite:///{db_path}", future=True, echo=False)
        Base.metadata.create_all(self._engine)

        self._Session: sessionmaker[Session] = sessionmaker(
            self._engine, expire_on_commit=False, class_=Session
        )

    # ------------------------------------------------------------------
    # Public API
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
        with self._Session() as session:
            session.add(
                _Message(channel_id=channel_id, message_json=json.dumps(message))
            )
            session.commit()

            if auto_trim:
                return self.trim_user_messages(channel_id)
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

    def trim_user_messages(self, channel_id: str) -> List[Dict[str, Any]]:
        """Trim old messages to keep only the most recent conversations.

        This removes the oldest messages while preserving the most recent
        conversations up to the user message limit.

        Args:
            channel_id (str): The channel to trim messages for.
            max_user_messages (int): Maximum number of user messages to keep.

        Returns:
            Updated message history after trimming.
        """
        if self._maximum_user_messages is None:
            return self.load_history(channel_id)

        with self._Session() as session:
            # Get all messages for the channel in chronological order
            stmt = (
                select(_Message)
                .where(_Message.channel_id == channel_id)  # type: ignore[arg-type]
                .order_by(_Message.created_at)  # type: ignore[arg-type]
            )
            rows = session.scalars(stmt).all()

            # Get the message IDs to keep
            messages_to_keep = self._get_message_ids_to_keep(
                rows, self._maximum_user_messages
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

    def _get_message_ids_to_keep(
        self, rows: Sequence[_Message], max_user_messages: int
    ) -> List[str]:
        """Get the message IDs to keep based on the user message limit.

        Args:
            rows: List of message rows in chronological order
            max_user_messages: Maximum number of user messages to keep

        Returns:
            List of message IDs to keep
        """
        user_message_count = 0
        messages_to_keep = []

        # Go through messages in reverse order (newest first) to keep recent ones
        for row in reversed(rows):
            try:
                message = json.loads(row.message_json)
                if message.get("role") == "user":
                    if user_message_count < max_user_messages:
                        user_message_count += 1
                        messages_to_keep.append(row.id)
                    else:
                        # Stop keeping messages once we hit the user message limit
                        break
                else:
                    # Keep non-user messages if we haven't hit the user limit yet
                    if user_message_count < max_user_messages:
                        messages_to_keep.append(row.id)
            except json.JSONDecodeError:
                continue

        return messages_to_keep
