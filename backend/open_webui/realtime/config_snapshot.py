"""Shared realtime config snapshot and payload helpers."""

from dataclasses import dataclass, fields
from typing import Any

from open_webui.config import AUDIO_RT_DEFAULT_IDLE_CALL_CHECKIN_PROMPT
from open_webui.models.config import Config
from open_webui.realtime.catalog import (
    get_effective_provider_model_id,
    is_realtime_model_id,
)


@dataclass(frozen=True)
class RealtimeConfigSnapshot:
    engine: str = ''
    api_key: str = ''
    api_base_url: str = 'https://api.openai.com/v1'
    models: tuple[str, ...] = ()
    voice: str = 'marin'
    vad_type: str = 'server_vad'
    server_vad_threshold: float = 0.5
    server_vad_silence_duration_ms: int = 500
    server_vad_prefix_padding_ms: int = 300
    semantic_vad_eagerness: str = 'auto'
    transcription_model: str = 'gpt-4o-transcribe'
    noise_reduction: str = 'near_field'
    max_response_output_tokens: str = ''
    context_enabled: bool = False
    context_recent_exchanges_limit: int = 10
    context_max_history_exchanges: int = 40
    context_max_history_bytes: int = 16000
    context_summarize: bool = False
    context_unanswered_last_user_turn: str = 'discard'
    context_summary_prompt: str = ''
    context_summary_max_size: int = 2000
    speed: float = 1.0
    transcription_prompt: str = ''
    vad_idle_timeout_ms: str = ''
    vad_create_response: bool = True
    vad_interrupt_response: bool = True
    session_timeout: int = 180
    idle_call_checkin_interval: int = 45
    idle_call_checkin_prompt: str = AUDIO_RT_DEFAULT_IDLE_CALL_CHECKIN_PROMPT
    truncation_strategy: str = 'auto'
    truncation_retention_ratio: float = 0.8
    truncation_token_limit: str = ''


# (snapshot_field, admin_field, config_dot_path). The admin_field is the wire name shared with the
# frontend admin form / RealtimeConfigForm; the dot_path is the persistent Config key (DEFAULT_CONFIG).
REALTIME_CONFIG_MAPPING: tuple[tuple[str, str, str], ...] = (
    ('engine', 'ENGINE', 'audio.realtime.engine'),
    ('api_key', 'API_KEY', 'audio.realtime.api_key'),
    ('api_base_url', 'API_BASE_URL', 'audio.realtime.api_base_url'),
    ('models', 'MODELS', 'audio.realtime.models'),
    ('voice', 'VOICE', 'audio.realtime.voice'),
    ('vad_type', 'VAD_TYPE', 'audio.realtime.vad_type'),
    ('server_vad_threshold', 'SERVER_VAD_THRESHOLD', 'audio.realtime.server_vad.threshold'),
    (
        'server_vad_silence_duration_ms',
        'SERVER_VAD_SILENCE_DURATION_MS',
        'audio.realtime.server_vad.silence_duration_ms',
    ),
    (
        'server_vad_prefix_padding_ms',
        'SERVER_VAD_PREFIX_PADDING_MS',
        'audio.realtime.server_vad.prefix_padding_ms',
    ),
    (
        'semantic_vad_eagerness',
        'SEMANTIC_VAD_EAGERNESS',
        'audio.realtime.semantic_vad.eagerness',
    ),
    ('transcription_model', 'TRANSCRIPTION_MODEL', 'audio.realtime.transcription_model'),
    ('noise_reduction', 'NOISE_REDUCTION', 'audio.realtime.noise_reduction'),
    (
        'max_response_output_tokens',
        'MAX_RESPONSE_OUTPUT_TOKENS',
        'audio.realtime.max_response_output_tokens',
    ),
    ('context_enabled', 'CONTEXT_ENABLED', 'audio.realtime.context.enabled'),
    (
        'context_recent_exchanges_limit',
        'CONTEXT_RECENT_EXCHANGES_LIMIT',
        'audio.realtime.context.recent_exchanges_limit',
    ),
    (
        'context_max_history_exchanges',
        'CONTEXT_MAX_HISTORY_EXCHANGES',
        'audio.realtime.context.max_history_exchanges',
    ),
    (
        'context_max_history_bytes',
        'CONTEXT_MAX_HISTORY_BYTES',
        'audio.realtime.context.max_history_bytes',
    ),
    ('context_summarize', 'CONTEXT_SUMMARIZE', 'audio.realtime.context.summarize'),
    (
        'context_unanswered_last_user_turn',
        'CONTEXT_UNANSWERED_LAST_USER_TURN',
        'audio.realtime.context.unanswered_last_user_turn',
    ),
    (
        'context_summary_prompt',
        'CONTEXT_SUMMARY_PROMPT',
        'audio.realtime.context.summary_prompt',
    ),
    (
        'context_summary_max_size',
        'CONTEXT_SUMMARY_MAX_SIZE',
        'audio.realtime.context.summary_max_size',
    ),
    ('speed', 'SPEED', 'audio.realtime.speed'),
    ('transcription_prompt', 'TRANSCRIPTION_PROMPT', 'audio.realtime.transcription_prompt'),
    ('vad_idle_timeout_ms', 'VAD_IDLE_TIMEOUT_MS', 'audio.realtime.vad_idle_timeout_ms'),
    ('vad_create_response', 'VAD_CREATE_RESPONSE', 'audio.realtime.vad_create_response'),
    (
        'vad_interrupt_response',
        'VAD_INTERRUPT_RESPONSE',
        'audio.realtime.vad_interrupt_response',
    ),
    ('session_timeout', 'SESSION_TIMEOUT', 'audio.realtime.session_timeout'),
    (
        'idle_call_checkin_interval',
        'IDLE_CALL_CHECKIN_INTERVAL',
        'audio.realtime.idle_call_checkin_interval',
    ),
    (
        'idle_call_checkin_prompt',
        'IDLE_CALL_CHECKIN_PROMPT',
        'audio.realtime.idle_call_checkin_prompt',
    ),
    ('truncation_strategy', 'TRUNCATION_STRATEGY', 'audio.realtime.truncation_strategy'),
    (
        'truncation_retention_ratio',
        'TRUNCATION_RETENTION_RATIO',
        'audio.realtime.truncation_retention_ratio',
    ),
    ('truncation_token_limit', 'TRUNCATION_TOKEN_LIMIT', 'audio.realtime.truncation_token_limit'),
)

_SNAPSHOT_FIELD_DEFAULTS = {field.name: field.default for field in fields(RealtimeConfigSnapshot)}

# admin wire field -> persistent dot-path, for Config.upsert on admin save.
REALTIME_CONFIG_KEYS = {admin_attr: dot_path for _, admin_attr, dot_path in REALTIME_CONFIG_MAPPING}


async def read_realtime_config() -> RealtimeConfigSnapshot:
    values = await Config.get_many(*(dot_path for _, _, dot_path in REALTIME_CONFIG_MAPPING))

    snapshot_values: dict[str, Any] = {}
    for snapshot_attr, _admin_attr, dot_path in REALTIME_CONFIG_MAPPING:
        value = values.get(dot_path, _SNAPSHOT_FIELD_DEFAULTS[snapshot_attr])
        if snapshot_attr == 'models':
            value = tuple(value or ())
        snapshot_values[snapshot_attr] = value

    return RealtimeConfigSnapshot(**snapshot_values)


async def build_realtime_admin_config(*, include_task_model_warning: bool = True) -> dict[str, Any]:
    snapshot = await read_realtime_config()
    payload = {
        admin_attr: list(getattr(snapshot, snapshot_attr))
        if snapshot_attr == 'models'
        else getattr(snapshot, snapshot_attr)
        for snapshot_attr, admin_attr, _dot_path in REALTIME_CONFIG_MAPPING
    }
    if include_task_model_warning:
        payload['TASK_MODEL_WARNING'] = await get_realtime_task_model_warning()
    return payload


async def apply_realtime_admin_config(realtime_form: Any) -> None:
    updates: dict[str, Any] = {}
    for snapshot_attr, admin_attr, dot_path in REALTIME_CONFIG_MAPPING:
        value = getattr(realtime_form, admin_attr)
        if snapshot_attr == 'models':
            value = list(value or [])
        updates[dot_path] = value
    await Config.upsert(updates)


async def build_realtime_client_defaults() -> dict[str, Any]:
    snapshot = await read_realtime_config()
    return {
        'voice': snapshot.voice,
        'vad_type': snapshot.vad_type,
        'server_vad_threshold': float(snapshot.server_vad_threshold),
        'server_vad_silence_duration_ms': int(snapshot.server_vad_silence_duration_ms),
        'server_vad_prefix_padding_ms': int(snapshot.server_vad_prefix_padding_ms),
        'semantic_vad_eagerness': str(snapshot.semantic_vad_eagerness),
        'noise_reduction': str(snapshot.noise_reduction),
        'speed': float(snapshot.speed),
        'vad_create_response': bool(snapshot.vad_create_response),
        'vad_interrupt_response': bool(snapshot.vad_interrupt_response),
        'session_timeout': int(snapshot.session_timeout),
        'idle_call_checkin_interval': int(snapshot.idle_call_checkin_interval),
    }


async def resolve_realtime_model_ids(models_state: Any = None) -> list[str]:
    snapshot = await read_realtime_config()
    if snapshot.models:
        return list(snapshot.models)

    if not models_state:
        return []

    items = models_state.items() if hasattr(models_state, 'items') else []
    seen: set[str] = set()
    realtime_model_ids: list[str] = []

    for _, model in items:
        model_id = get_effective_provider_model_id(model)
        if not is_realtime_model_id(model_id):
            continue

        normalized_model_id = model_id.lower()
        if normalized_model_id in seen:
            continue

        seen.add(normalized_model_id)
        realtime_model_ids.append(model_id)

    return realtime_model_ids


async def get_realtime_task_model_warning() -> bool:
    values = await Config.get_many('audio.realtime.engine', 'task.model.external')
    return values.get('audio.realtime.engine', '') == 'openai' and not values.get('task.model.external', '')
