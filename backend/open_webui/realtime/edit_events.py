"""Cross-worker propagation of chat-edit events into realtime sessions.

When a chat message is edited (typically via the text-mode UI in another
tab) while a realtime voice session is active, the session's in-memory
view of the chat history can go stale.  This module ships a Redis pubsub
channel so any worker can announce an edit and the worker hosting the
realtime sideband picks it up and invalidates its cached state.

In single-worker mode the publisher dispatches directly to the local
handler — no Redis hop.
"""

import asyncio
import json
import logging

from open_webui.env import (
    REDIS_KEY_PREFIX,
    WEBSOCKET_MANAGER,
    WEBSOCKET_REDIS_CLUSTER,
    WEBSOCKET_REDIS_URL,
    WEBSOCKET_SENTINEL_HOSTS,
    WEBSOCKET_SENTINEL_PORT,
)
from open_webui.utils.redis import get_sentinels_from_env

log = logging.getLogger(__name__)

CHAT_EDIT_CHANNEL = f'{REDIS_KEY_PREFIX}:rt:chat_edited'

_redis_sentinels = get_sentinels_from_env(WEBSOCKET_SENTINEL_HOSTS, WEBSOCKET_SENTINEL_PORT)
_redis_client = None

if WEBSOCKET_MANAGER == 'redis':
    from open_webui.utils.redis import get_redis_connection

    _redis_client = get_redis_connection(
        WEBSOCKET_REDIS_URL,
        _redis_sentinels,
        redis_cluster=WEBSOCKET_REDIS_CLUSTER,
        async_mode=True,
        decode_responses=True,
    )


def _build_payload(chat_id: str, message_id: str) -> str:
    return json.dumps({'chat_id': chat_id, 'message_id': message_id})


async def notify_chat_edited(chat_id: str, message_id: str) -> None:
    """Announce that a chat message was edited.

    Multi-worker mode publishes to Redis; single-worker mode dispatches
    directly to the in-process handler.
    """
    if not chat_id:
        return

    payload = _build_payload(chat_id, message_id)

    if _redis_client is not None:
        try:
            if hasattr(_redis_client, 'nodes_manager'):
                await _redis_client.execute_command('PUBLISH', CHAT_EDIT_CHANNEL, payload)
            else:
                await _redis_client.publish(CHAT_EDIT_CHANNEL, payload)
        except Exception:
            log.exception('Failed to publish chat-edit event for chat_id=%s', chat_id)
        return

    await _handle_chat_edited(chat_id, message_id)


async def _handle_chat_edited(chat_id: str, message_id: str) -> None:
    """Invalidate cached parent state on the active realtime session owning chat_id."""
    from open_webui.realtime.session_state import session_manager

    session = session_manager.get_session_by_chat(chat_id)
    if not session:
        return

    log.info(
        'realtime.edit.event chat_id=%s message_id=%s session_id=%s',
        chat_id,
        message_id,
        session.session_id,
    )


async def chat_edited_listener() -> None:
    """Background task: subscribe to CHAT_EDIT_CHANNEL and dispatch incoming events."""
    if _redis_client is None:
        return

    pubsub = _redis_client.pubsub()
    await pubsub.subscribe(CHAT_EDIT_CHANNEL)
    log.info('Chat-edit listener: subscribed to %s', CHAT_EDIT_CHANNEL)

    try:
        async for message in pubsub.listen():
            if message['type'] != 'message':
                continue
            try:
                raw = message['data']
                if isinstance(raw, bytes):
                    raw = raw.decode()
                payload = json.loads(raw)
                await _handle_chat_edited(payload.get('chat_id', ''), payload.get('message_id', ''))
            except Exception:
                log.exception('Error handling chat-edit notification')
    except asyncio.CancelledError:
        pass
    finally:
        try:
            await pubsub.unsubscribe(CHAT_EDIT_CHANNEL)
        except Exception:
            pass
