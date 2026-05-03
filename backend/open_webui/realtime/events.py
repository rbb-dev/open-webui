"""Shared event emission helpers for the realtime package."""

import asyncio
import logging
from typing import Optional

log = logging.getLogger(__name__)

_background_tasks: set[asyncio.Task] = set()


async def _bg(coro) -> None:
    """Run a coroutine with exception logging (used by fire_and_forget)."""
    try:
        await coro
    except Exception:
        log.exception('Background task failed')


def fire_and_forget(coro) -> None:
    """Schedule a coroutine as a background task with GC protection."""
    try:
        task = asyncio.create_task(_bg(coro))
    except RuntimeError:
        log.warning('fire_and_forget called with no running event loop')
        return
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


async def emit_to_user(sio, user_id: str, chat_id: str, message_id: str, event_data: dict) -> None:
    """Emit a Socket.IO event to a user's room."""
    await sio.emit(
        'events',
        {
            'chat_id': chat_id,
            'message_id': message_id,
            'data': event_data,
        },
        room=f'user:{user_id}',
    )


async def emit_status(sio, user_id: str, chat_id: str, call_id: str, description: str, done: bool) -> None:
    """Emit a status event to the user's room."""
    await emit_to_user(
        sio,
        user_id,
        chat_id,
        '',
        {
            'type': 'status',
            'data': {
                'description': description,
                'done': done,
                'callId': call_id,
            },
        },
    )


def realtime_notification_meta(call_id: Optional[str] = None) -> dict:
    """Build notification metadata for realtime events."""
    notification = {'suppress': True, 'realtime': True}
    if call_id:
        notification['callId'] = call_id
    return {'notification': notification}
