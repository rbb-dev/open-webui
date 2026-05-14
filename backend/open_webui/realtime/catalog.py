"""Realtime-capable model discovery and provider model helpers."""

import logging
from typing import Any

from open_webui.realtime.constants import (
    ALL_REALTIME_VOICES,
    get_realtime_voice_ids_for_model,
)

log = logging.getLogger(__name__)


def is_realtime_model_id(model_id: str) -> bool:
    return 'realtime' in (model_id or '').lower()


def get_effective_provider_model_id(model: Any) -> str:
    if not isinstance(model, dict):
        return ''

    info = model.get('info')
    if isinstance(info, dict):
        base_model_id = info.get('base_model_id')
        if isinstance(base_model_id, str) and base_model_id:
            return base_model_id

    base_model_id = model.get('base_model_id')
    if isinstance(base_model_id, str) and base_model_id:
        return base_model_id

    model_id = model.get('id')
    return model_id if isinstance(model_id, str) else ''


def model_uses_realtime(
    request: Any,
    model: Any,
    *,
    require_realtime_engine: bool = True,
) -> bool:
    if require_realtime_engine and getattr(request.app.state.config, 'AUDIO_RT_ENGINE', '') != 'openai':
        return False

    if require_realtime_engine and not getattr(request.app.state.config, 'AUDIO_RT_API_KEY', ''):
        return False

    model_id = get_effective_provider_model_id(model)
    if not is_realtime_model_id(model_id):
        return False

    whitelist = list(getattr(request.app.state.config, 'AUDIO_RT_MODELS', []))
    if not whitelist:
        return True

    whitelist_set = {whitelisted_model_id.lower() for whitelisted_model_id in whitelist}
    return model_id.lower() in whitelist_set


def build_voice_capabilities(model_ids: list[str]) -> dict[str, Any]:
    normalized_ids = [model_id for model_id in model_ids if is_realtime_model_id(model_id)]
    supported_voices = (
        sorted({voice_id for model_id in normalized_ids for voice_id in get_realtime_voice_ids_for_model(model_id)})
        if normalized_ids
        else list(ALL_REALTIME_VOICES)
    )

    return {
        'voices': [{'id': voice_id, 'name': voice_id} for voice_id in supported_voices],
        'voices_by_model': {model_id: list(get_realtime_voice_ids_for_model(model_id)) for model_id in normalized_ids},
    }
