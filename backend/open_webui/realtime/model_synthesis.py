"""Realtime model list synthesis.

Fetches the realtime model catalog from the configured OpenAI-compatible
realtime endpoint, filters via is_realtime_model_id, applies the
admin whitelist, and synthesizes display fields expected by the rest of
the OWUI model pipeline (name, owned_by, connection_type).

Used by utils.models.get_all_base_models to fold realtime-only model
ids into the global model registry.
"""

import hashlib
import logging

from aiocache import cached
from fastapi import Request

from open_webui.env import MODELS_CACHE_TTL
from open_webui.models.config import Config
from open_webui.models.users import UserModel
from open_webui.realtime.catalog import is_realtime_model_id

log = logging.getLogger(__name__)


# Last good list and the config key it was fetched under; served only for that same config.
_rt_models_fallback: tuple[str, list[dict]] = ('', [])


def _fallback_for(fallback_key: str) -> list[dict]:
    key, models = _rt_models_fallback
    return list(models) if key == fallback_key else []


def _realtime_models_cache_key(_func, rt_base_url, rt_api_key, rt_whitelist, user=None):
    key_digest = hashlib.sha256(rt_api_key.encode()).hexdigest()[:16]
    return f'realtime_models:{rt_base_url}:{key_digest}:{",".join(rt_whitelist)}'


@cached(ttl=MODELS_CACHE_TTL, key_builder=_realtime_models_cache_key)
async def _fetch_realtime_models_cached(rt_base_url, rt_api_key, rt_whitelist, user=None) -> list[dict]:
    global _rt_models_fallback
    fallback_key = _realtime_models_cache_key(None, rt_base_url, rt_api_key, rt_whitelist)
    try:
        from open_webui.routers.openai import send_get_request

        data = await send_get_request(url=f'{rt_base_url}/models', key=rt_api_key, user=user)
        if not isinstance(data, dict):
            # send_get_request returns None on a connection error or timeout.
            return _fallback_for(fallback_key)
        all_models = data.get('data', [])
        rt_models = [
            {
                **m,
                'name': m.get('name', m.get('id', '')),
                'owned_by': m.get('owned_by', 'openai'),
                'connection_type': m.get('connection_type', 'external'),
            }
            for m in all_models
            if is_realtime_model_id(m.get('id', '')) and (not rt_whitelist or m.get('id') in rt_whitelist)
        ]
        if rt_models:
            _rt_models_fallback = (fallback_key, rt_models)
        return rt_models
    except Exception as e:
        log.warning(f'Failed to fetch realtime models: {e}')
        return _fallback_for(fallback_key)


async def fetch_realtime_models(request: Request, user: UserModel = None) -> list[dict]:
    try:
        rt_config = await Config.get_many(
            'audio.realtime.api_base_url',
            'audio.realtime.api_key',
            'audio.realtime.models',
        )
    except Exception as e:
        log.warning(f'Failed to read realtime config: {e}')
        return []
    rt_base_url = str(rt_config.get('audio.realtime.api_base_url', '')).rstrip('/')
    rt_api_key = str(rt_config.get('audio.realtime.api_key', ''))
    if not rt_base_url or not rt_api_key:
        return []
    rt_whitelist = tuple(rt_config.get('audio.realtime.models', []) or [])
    return await _fetch_realtime_models_cached(rt_base_url, rt_api_key, rt_whitelist, user=user)
