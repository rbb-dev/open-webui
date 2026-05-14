"""Client-facing realtime defaults and capability data."""

from typing import Any

from open_webui.models.config import Config
from open_webui.realtime.catalog import build_voice_capabilities
from open_webui.realtime.config_snapshot import (
    build_realtime_client_defaults,
    resolve_realtime_model_ids,
)
from open_webui.realtime.constants import (
    REALTIME_NOISE_REDUCTION_TYPES,
    REALTIME_SEMANTIC_VAD_EAGERNESS,
    REALTIME_VAD_TYPES,
)


async def build_realtime_client_config(request: Any) -> dict[str, Any]:
    engine_and_key = await Config.get_many('audio.realtime.engine', 'audio.realtime.api_key')
    enabled = engine_and_key.get('audio.realtime.engine', '') == 'openai' and bool(
        engine_and_key.get('audio.realtime.api_key', '')
    )

    model_ids = await resolve_realtime_model_ids(
        getattr(request.app.state, 'MODELS', None),
    )

    return {
        'enabled': enabled,
        'defaults': await build_realtime_client_defaults(),
        'capabilities': {
            'models': model_ids,
            **build_voice_capabilities(model_ids),
            'vad_types': list(REALTIME_VAD_TYPES),
            'noise_reduction': list(REALTIME_NOISE_REDUCTION_TYPES),
            'semantic_vad_eagerness': list(REALTIME_SEMANTIC_VAD_EAGERNESS),
        },
    }
