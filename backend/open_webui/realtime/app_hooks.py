"""Realtime startup hooks owned by the realtime subsystem."""

import asyncio
import logging

from fastapi import FastAPI

from open_webui.env import WEBSOCKET_MANAGER
from open_webui.realtime.session_service import orphan_sweep_loop, session_manager

# sio is imported lazily inside each function to avoid a circular import with
# open_webui.socket.main, which imports this module at startup.

log = logging.getLogger(__name__)


async def _pending_text_listener() -> None:
    """Multi-worker pub/sub listener for pending text notifications."""
    from open_webui.realtime.pending_store import (
        _NOTIFY_CHANNEL,
        _redis_client,
    )
    from open_webui.socket.main import sio

    if _redis_client is None:
        return

    pubsub = _redis_client.pubsub()
    await pubsub.subscribe(_NOTIFY_CHANNEL)
    log.info('Pending text listener: subscribed to %s', _NOTIFY_CHANNEL)

    try:
        async for message in pubsub.listen():
            if message['type'] != 'message':
                continue
            try:
                session_id = message['data'].decode() if isinstance(message['data'], bytes) else message['data']
                session = session_manager.get_session(session_id)
                if (
                    not session
                    or not session.is_ready
                    or session.assistant_responding
                    or session.tool_executing
                    or not session.ws
                ):
                    continue
                from open_webui.realtime.sideband import _flush_pending_to_ws

                await _flush_pending_to_ws(session, [], 'notified_pending', sio=sio, drain_redis=True)
            except Exception:
                log.exception('Error handling pending text notification')
    except asyncio.CancelledError:
        pass
    finally:
        try:
            await pubsub.unsubscribe(_NOTIFY_CHANNEL)
        except Exception:
            pass


def start_realtime_background_tasks(app: FastAPI) -> list[asyncio.Task]:
    from open_webui.realtime.edit_events import chat_edited_listener
    from open_webui.realtime.pending_store import _redis_client as _pending_redis
    from open_webui.socket.main import sio

    # If pending_store decided WEBSOCKET_MANAGER == 'redis' at import time but
    # somehow ended up without a redis client, the listeners would silently
    # subscribe to nothing. Surface that mismatch loudly during startup.
    if WEBSOCKET_MANAGER == 'redis' and _pending_redis is None:
        log.error(
            'WEBSOCKET_MANAGER=redis but pending_store has no Redis client. '
            'Pending text + chat-edit listeners will not receive events. '
            'Check WEBSOCKET_REDIS_URL / sentinel config.',
        )

    tasks = [asyncio.create_task(orphan_sweep_loop(sio, app.state.config))]
    if WEBSOCKET_MANAGER == 'redis':
        tasks.append(asyncio.create_task(_pending_text_listener()))
        tasks.append(asyncio.create_task(chat_edited_listener()))
    app.state.realtime_tasks = tasks
    return tasks


async def stop_realtime_background_tasks(app: FastAPI) -> None:
    """Cancel and await all realtime background tasks; called from app shutdown.

    Also drains in-flight session-end background tasks (title / tag /
    follow-up generation) so they aren't lost mid-write.
    """
    tasks: list[asyncio.Task] = getattr(app.state, 'realtime_tasks', []) or []
    for task in tasks:
        if not task.done():
            task.cancel()
    for task in tasks:
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception:
            log.exception('Realtime background task raised during shutdown')
    app.state.realtime_tasks = []

    await session_manager.drain_end_tasks()


async def handle_socket_disconnect(sid: str) -> None:
    """Tear down realtime state for a disconnecting socket. Called from socket/main.py."""
    from open_webui.socket.main import sio

    rt_session = session_manager.get_session(sid)
    if not rt_session:
        return
    await session_manager.teardown_session(sid, reason='disconnect', sio=sio)

    from open_webui.realtime.ownership import get_records_by_sid
    from open_webui.realtime.ownership import release as release_ownership

    for record in get_records_by_sid(sid):
        await release_ownership(record.voice_session_id)
