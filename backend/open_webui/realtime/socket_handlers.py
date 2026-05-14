"""Socket.IO registration helpers for realtime voice events."""

import logging
from functools import partial

from pydantic import ValidationError

from open_webui.models.chats import Chats
from open_webui.realtime.contracts import (
    RealtimeCallPayload,
    RealtimeSettingsUpdatePayload,
    RealtimeStartPayload,
    realtime_event_payload,
)
from open_webui.realtime.session_service import (
    session_manager,
    start_realtime_session,
)
from open_webui.realtime.session_state import (
    cancel_checkin_timer,
    record_user_activity,
    restart_checkin_timer_only,
)
from open_webui.utils.access_control import get_permissions, has_permission

log = logging.getLogger(__name__)


SETTING_TO_PERMISSION = {
    'voice': 'voice',
    'speed': 'speed',
    'vad_type': 'vad',
    'noise_reduction': 'noise_reduction',
    'max_tokens': 'max_tokens',
    'context': 'context',
}


def _get_matching_session(sid: str, call_id: str, voice_session_id: str = None, generation: int = None):
    session = session_manager.get_session(sid)
    if not session or session.call_id != call_id:
        return None
    if voice_session_id and session.voice_session_id and session.voice_session_id != voice_session_id:
        log.warning(
            'Session mismatch: voice_session_id expected=%s got=%s', session.voice_session_id, voice_session_id
        )
        return None
    if generation is not None and session.generation != generation:
        log.warning('Session mismatch: generation expected=%s got=%s', session.generation, generation)
        return None
    return session


async def realtime_start(  # noqa: C901 — sequential pre-flight: validate payload, check chat.call permission, check chat/channel ownership, then start the session. Each gate is short and decomposing would require threading sid/payload/sio/user through helper signatures.
    sio,
    session_pool,
    get_socket_app,
    get_event_emitter,
    get_event_call,
    sid,
    data,
) -> None:
    user = session_pool.get(sid)
    if not user:
        return
    payload_data = data or {}

    try:
        payload = RealtimeStartPayload.model_validate(payload_data)
    except ValidationError:
        await sio.emit(
            'realtime:ended',
            realtime_event_payload(
                call_id=(payload_data or {}).get('callId'),
                reason='invalid_request',
                message='Open WebUI: Invalid realtime start payload.',
            ),
            room=sid,
        )
        return

    app = get_socket_app(sid)
    app_config = getattr(getattr(app, 'state', None), 'config', None) if app else None

    if user.get('role') != 'admin' and not await has_permission(
        user['id'], 'chat.call', getattr(app_config, 'USER_PERMISSIONS', {})
    ):
        await sio.emit(
            'realtime:ended',
            realtime_event_payload(call_id=payload.callId, reason='forbidden'),
            room=sid,
        )
        return

    model_id = payload.modelId
    chat_id = payload.chatId
    if not model_id or not chat_id:
        log.warning(
            'realtime:start rejected: missing model_id=%r chat_id=%r sid=%s',
            model_id,
            chat_id,
            sid,
        )
        await sio.emit(
            'realtime:ended',
            realtime_event_payload(call_id=payload.callId, reason='missing_params'),
            room=sid,
        )
        return

    if chat_id and chat_id.startswith('channel:') and user.get('role') != 'admin':
        channel_id = chat_id.split(':', 1)[1]
        from open_webui.models.channels import Channels

        if not channel_id or not await Channels.is_user_channel_member(channel_id, user['id']):
            log.warning(
                'realtime:start rejected: channel membership check failed sid=%s user=%s channel=%s',
                sid,
                user['id'],
                channel_id,
            )
            await sio.emit(
                'realtime:ended',
                realtime_event_payload(call_id=payload.callId, reason='forbidden'),
                room=sid,
            )
            return
    elif (
        chat_id
        and not chat_id.startswith('local:')
        and not chat_id.startswith('channel:')
        and user.get('role') != 'admin'
    ):
        if not await Chats.is_chat_owner(chat_id, user['id']):
            log.warning(
                'realtime:start rejected: chat ownership check failed sid=%s user=%s chat=%s',
                sid,
                user['id'],
                chat_id,
            )
            await sio.emit(
                'realtime:ended',
                realtime_event_payload(call_id=payload.callId, reason='forbidden'),
                room=sid,
            )
            return

    try:
        await start_realtime_session(
            sid,
            user,
            model_id,
            chat_id,
            payload.callId,
            sio,
            app_config,
            event_emitter_factory=get_event_emitter,
            event_caller_factory=get_event_call,
            app=app,
            tool_ids=payload.toolIds,
            tool_servers=payload.toolServers,
            features=payload.features,
            terminal_id=payload.terminalId,
            folder_id=payload.folderId or '',
        )
    except Exception:
        log.exception('realtime:start crashed for sid=%s', sid)
        await sio.emit(
            'realtime:ended',
            realtime_event_payload(
                call_id=payload.callId,
                reason='error',
                message='Open WebUI: Failed to start realtime voice session.',
            ),
            room=sid,
        )


async def realtime_stop(sio, sid, data=None) -> None:
    payload_data = data or {}
    try:
        payload = RealtimeCallPayload.model_validate(payload_data)
    except ValidationError:
        return
    session = _get_matching_session(sid, payload.callId, payload.voiceSessionId, payload.generation)
    if not session:
        return
    await session_manager.teardown_session(sid, reason='user', sio=sio)
    await sio.emit(
        'realtime:ended',
        realtime_event_payload(
            call_id=payload.callId,
            voice_session_id=session.voice_session_id,
            generation=session.generation,
            reason='user',
        ),
        room=sid,
    )


async def realtime_commit_audio(sio, sid, data=None) -> None:
    try:
        payload = RealtimeCallPayload.model_validate(data or {})
    except ValidationError:
        return
    session = _get_matching_session(sid, payload.callId, payload.voiceSessionId, payload.generation)
    if not session:
        return
    await record_user_activity(session, sio)
    await session.send_event('input_audio_buffer.commit', context='ptt_commit')
    await session.send_event('response.create', context='ptt_response')


async def realtime_clear_audio(sid, data=None) -> None:
    try:
        payload = RealtimeCallPayload.model_validate(data or {})
    except ValidationError:
        return
    session = _get_matching_session(sid, payload.callId, payload.voiceSessionId, payload.generation)
    if not session:
        return
    await session.send_event('input_audio_buffer.clear', context='ptt_clear')


async def realtime_stop_checkin(sid, data=None) -> None:
    try:
        payload = RealtimeCallPayload.model_validate(data or {})
    except ValidationError:
        return
    session = _get_matching_session(sid, payload.callId, payload.voiceSessionId, payload.generation)
    if not session:
        return
    session.checkin_suppressed = True
    await cancel_checkin_timer(session)


async def realtime_start_checkin(sio, sid, data=None) -> None:
    try:
        payload = RealtimeCallPayload.model_validate(data or {})
    except ValidationError:
        return
    session = _get_matching_session(sid, payload.callId, payload.voiceSessionId, payload.generation)
    if not session:
        return
    session.checkin_suppressed = False
    await restart_checkin_timer_only(session, sio)


async def realtime_settings_update(sio, session_pool, get_socket_app, sid, data) -> None:
    try:
        payload = RealtimeSettingsUpdatePayload.model_validate(data or {})
    except ValidationError:
        return
    session = _get_matching_session(sid, payload.callId, payload.voiceSessionId, payload.generation)
    if not session:
        return

    setting = payload.setting
    value = payload.value
    if not setting:
        return

    perm_key = SETTING_TO_PERMISSION.get(setting)
    if not perm_key:
        return

    user = session_pool.get(sid)
    if user and user.get('role') != 'admin':
        app = get_socket_app(sid)
        app_config = getattr(getattr(app, 'state', None), 'config', None)
        user_perms = await get_permissions(user['id'], getattr(app_config, 'USER_PERMISSIONS', {}))
        rt_perms = user_perms.get('realtime', {}).get('settings', {})
        if not rt_perms.get(perm_key, False):
            return

    if session.assistant_responding:
        session.pending_settings_updates.append({'setting': payload.setting, 'value': payload.value})
        return

    await session.apply_setting(setting, value, sio=sio)


async def realtime_cancel(sio, sid, data=None) -> None:
    try:
        payload = RealtimeCallPayload.model_validate(data or {})
    except ValidationError:
        return
    session = _get_matching_session(sid, payload.callId, payload.voiceSessionId, payload.generation)
    if not session:
        return
    active_response_id = session.active_response_id
    if not active_response_id:
        return
    if payload.responseId and payload.responseId != active_response_id:
        return
    if session.cancel_requested_response_id == active_response_id:
        return
    session.cancel_requested_response_id = active_response_id
    await sio.emit(
        'realtime:response_state',
        realtime_event_payload(
            call_id=session.call_id,
            voice_session_id=session.voice_session_id,
            generation=session.generation,
            responseId=active_response_id,
            phase='cancel_requested',
        ),
        room=sid,
    )
    await session.send_event(
        'response.cancel',
        context=f'response_cancel:{active_response_id}',
    )


async def realtime_chat_edit(session_pool, sid, data=None) -> None:
    """Frontend announces a chat message edit; propagate to the worker
    hosting that chat's realtime session so it can drop stale parent
    state before the next assistant completion."""
    payload_data = data or {}
    chat_id = str(payload_data.get('chatId') or '').strip()
    message_id = str(payload_data.get('messageId') or '').strip()
    if not chat_id or not message_id:
        return

    user = session_pool.get(sid)
    if user and chat_id.startswith('channel:') and user.get('role') != 'admin':
        channel_id = chat_id.split(':', 1)[1]
        from open_webui.models.channels import Channels

        if not channel_id or not await Channels.is_user_channel_member(channel_id, user['id']):
            return
    elif (
        user
        and not chat_id.startswith('local:')
        and not chat_id.startswith('channel:')
        and user.get('role') != 'admin'
    ):
        if not await Chats.is_chat_owner(chat_id, user['id']):
            return

    from open_webui.realtime.edit_events import notify_chat_edited

    await notify_chat_edited(chat_id, message_id)


def register_realtime_socket_handlers(
    *,
    sio,
    session_pool,
    get_socket_app,
    get_event_emitter,
    get_event_call,
) -> None:
    sio.on('realtime:start', partial(
        realtime_start, sio, session_pool, get_socket_app, get_event_emitter, get_event_call,
    ))
    sio.on('realtime:stop', partial(realtime_stop, sio))
    sio.on('realtime:commit_audio', partial(realtime_commit_audio, sio))
    sio.on('realtime:clear_audio', realtime_clear_audio)
    sio.on('realtime:stop_checkin', realtime_stop_checkin)
    sio.on('realtime:start_checkin', partial(realtime_start_checkin, sio))
    sio.on('realtime:settings:update', partial(realtime_settings_update, sio, session_pool, get_socket_app))
    sio.on('realtime:cancel', partial(realtime_cancel, sio))
    sio.on('realtime:chat_edit', partial(realtime_chat_edit, session_pool))
