"""Per-chat write lock for realtime persistence.

Serializes chat-history mutations on the same chat across realtime turn
lifecycle functions.  In multi-worker deployments the lock is backed by
Redis (via open_webui.socket.utils.RedisLock) so different workers
contend on the same lock key; in single-worker / in-memory mode the lock
is a plain asyncio.Lock.

Reuses the same dual-mode pattern as
open_webui.realtime.ownership to keep one cohesive locking story
across the realtime subsystem.

A background watchdog task renews the Redis lock periodically (every
WEBSOCKET_REDIS_LOCK_TIMEOUT / 3 seconds) so long-running voice
turns do not lose the lock mid-persist.  If the lock cannot be renewed
(network blip, another worker stole it), the watchdog sets
status.lost and logs at ERROR level — callers operating inside the
async with block are expected to check status.lost.is_set() at
their own commit boundaries and abort the in-flight write rather than
risk a split-brain mutation.

Note on memory growth: the single-worker fallback dict
_memory_chat_locks grows unbounded across all chats touched.  This
mirrors the same known limitation in
open_webui.realtime.ownership (_memory_locks); both will be
addressed if/when a shared eviction pass is added.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from open_webui.env import (
    REDIS_KEY_PREFIX,
    WEBSOCKET_MANAGER,
    WEBSOCKET_REDIS_CLUSTER,
    WEBSOCKET_REDIS_LOCK_TIMEOUT,
    WEBSOCKET_REDIS_URL,
    WEBSOCKET_SENTINEL_HOSTS,
    WEBSOCKET_SENTINEL_PORT,
)
from open_webui.utils.redis import get_sentinels_from_env

log = logging.getLogger(__name__)


_redis_sentinels = get_sentinels_from_env(WEBSOCKET_SENTINEL_HOSTS, WEBSOCKET_SENTINEL_PORT)

# In-memory locks for single-worker / non-Redis deployments.
# Same known unbounded-growth pattern as realtime.ownership._memory_locks.
_memory_chat_locks: dict[str, asyncio.Lock] = {}


class ChatLockUnavailable(RuntimeError):
    """Raised when the per-chat lock cannot be acquired."""


@dataclass
class ChatLockStatus:
    """Lock status object yielded to the caller of chat_write_lock.

    lost is set if the Redis watchdog cannot renew the underlying lock.
    Callers operating inside the async with block should check
    status.lost.is_set() at commit boundaries and abort the write to
    avoid split-brain mutations.
    """

    lost: asyncio.Event = field(default_factory=asyncio.Event)


def _build_redis_lock(chat_id: str):
    from open_webui.socket.utils import RedisLock

    return RedisLock(
        redis_url=WEBSOCKET_REDIS_URL,
        lock_name=f'{REDIS_KEY_PREFIX}:rt_chat_write:{chat_id}',
        timeout_secs=WEBSOCKET_REDIS_LOCK_TIMEOUT,
        redis_sentinels=_redis_sentinels,
        redis_cluster=WEBSOCKET_REDIS_CLUSTER,
    )


async def _renewal_loop(lock, interval_secs: float, status: ChatLockStatus) -> None:
    """Periodically refresh the Redis lock TTL while held.

    Sets status.lost on the first failed or errored renewal so the
    caller can detect ownership loss at a safe checkpoint.  Any
    exception raised by RedisLock.renew_lock (network blip, Redis
    down) is treated as ownership loss — the watchdog never silently
    dies without notifying the caller.
    """
    try:
        while True:
            await asyncio.sleep(interval_secs)
            try:
                renewed = lock.renew_lock()
            except Exception as exc:
                log.error(
                    'realtime.chat_lock.renewal_error',
                    extra={'lock_name': lock.lock_name, 'error': str(exc)},
                )
                status.lost.set()
                return
            if not renewed:
                log.error(
                    'realtime.chat_lock.renewal_failed',
                    extra={'lock_name': lock.lock_name},
                )
                status.lost.set()
                return
    except asyncio.CancelledError:
        pass


@asynccontextmanager
async def chat_write_lock(chat_id: str):
    """Async context manager serializing chat-history writes for one chat.

    Multi-worker mode acquires a Redis-backed lock and runs a watchdog task
    that renews the lock every WEBSOCKET_REDIS_LOCK_TIMEOUT / 3 seconds.
    Single-worker mode falls back to asyncio.Lock indexed by
    chat_id.

    Yields a ChatLockStatus whose lost event is set if Redis
    renewal fails.  Callers must check status.lost.is_set() at commit
    boundaries to avoid split-brain writes.  Single-worker mode never
    sets the event.
    """
    if not chat_id:
        raise ChatLockUnavailable('chat_write_lock requires a non-empty chat_id')

    status = ChatLockStatus()

    if WEBSOCKET_MANAGER == 'redis':
        lock = _build_redis_lock(chat_id)
        if not lock.aquire_lock():
            raise ChatLockUnavailable(f'Could not acquire chat lock for {chat_id}')
        # Renew at TTL/3 so a transient slow operation does not lose ownership.
        renewal_interval = max(1.0, WEBSOCKET_REDIS_LOCK_TIMEOUT / 3.0)
        renewal_task = asyncio.create_task(_renewal_loop(lock, renewal_interval, status))
        try:
            yield status
        finally:
            renewal_task.cancel()
            try:
                await renewal_task
            except asyncio.CancelledError:
                pass
            try:
                lock.release_lock()
            except Exception as exc:
                log.warning(
                    'realtime.chat_lock.release_failed',
                    extra={'lock_name': lock.lock_name, 'error': str(exc)},
                )
    else:
        lock = _memory_chat_locks.setdefault(chat_id, asyncio.Lock())
        async with lock:
            yield status
