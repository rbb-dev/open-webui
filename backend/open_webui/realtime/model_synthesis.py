"""Realtime model list synthesis.

Fetches the realtime model catalog from the configured OpenAI-compatible
realtime endpoint, filters via is_realtime_model_id, applies the
admin whitelist, and synthesizes display fields expected by the rest of
the OWUI model pipeline (name, owned_by, connection_type).

Used by utils.models.get_all_base_models to fold realtime-only model
ids into the global model registry.
"""

import logging

from aiocache import cached
from fastapi import Request

from open_webui.models.users import UserModel
from open_webui.realtime.catalog import is_realtime_model_id

log = logging.getLogger(__name__)


_rt_models_fallback: list[dict] = []


@cached(ttl=3600, key='realtime_models')
async def _fetch_realtime_models_cached(request: Request, user: UserModel = None) -> list[dict]:
    global _rt_models_fallback
    try:
        rt_base_url = str(getattr(request.app.state.config, 'AUDIO_RT_API_BASE_URL', '')).rstrip('/')
        rt_api_key = str(getattr(request.app.state.config, 'AUDIO_RT_API_KEY', ''))
        rt_whitelist = list(getattr(request.app.state.config, 'AUDIO_RT_MODELS', []) or [])
        if not rt_base_url or not rt_api_key:
            return []

        from open_webui.routers.openai import send_get_request

        data = await send_get_request(url=f'{rt_base_url}/models', key=rt_api_key, user=user)
        all_models = data.get('data', []) if isinstance(data, dict) else []
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
            _rt_models_fallback = rt_models
        return rt_models
    except Exception as e:
        log.warning(f'Failed to fetch realtime models: {e}')
        return list(_rt_models_fallback)


async def fetch_realtime_models(request: Request, user: UserModel = None) -> list[dict]:
    return await _fetch_realtime_models_cached(request, user=user)
