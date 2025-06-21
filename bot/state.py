from __future__ import annotations

import json
import os
import uuid
from datetime import datetime
from typing import Any, Dict, List

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

    def __init__(self, db_path: str):
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

    def append(self, channel_id: str, message: Dict[str, Any]) -> None:
        """Persist a single message for a channel."""
        with self._Session() as session:
            session.add(
                _Message(channel_id=channel_id, message_json=json.dumps(message))
            )
            session.commit()

    def reset(self, channel_id: str | None = None) -> None:
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
