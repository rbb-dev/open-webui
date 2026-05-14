"""Realtime sideband protocol runtime."""

import asyncio
import json
import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar
from uuid import uuid4

from open_webui.realtime.chat_lock import chat_write_lock
from open_webui.realtime.chat_sync import (
    _upsert_message_no_advance,
    build_assistant_output_message_item,
    create_realtime_text_turn,
    create_voice_turn_messages,
    emit_turn_output,
    ensure_realtime_voice_turn,
    ensure_trailing_assistant_output_message,
    maybe_gc_turn,
    persist_turn_output,
    remove_orphan_turn_messages,
    resolve_realtime_voice_assistant_reply_message,
)
from open_webui.realtime.constants import ASSISTANT_LISTENING_PLACEHOLDER
from open_webui.realtime.contracts import RealtimeTurnHandoff
from open_webui.realtime.events import emit_status, emit_to_user, fire_and_forget
from open_webui.realtime.pending_store import pop_one_pending_text
from open_webui.realtime.session_state import (
    RealtimeSession,
    cancel_all_timers,
    cancel_checkin_timer,
    end_session,
    maybe_emit_ready,
    record_user_activity,
    reset_bootstrap_flags,
    restart_all_timers,
    restart_checkin_timer_only,
)
from open_webui.realtime.turn_state import VoiceTurn

log = logging.getLogger(__name__)


class _SidebandClosed(Exception):
    def __init__(self, code: int):
        super().__init__(f'Sideband closed with code {code}')
        self.code = code


async def _flush_pending_to_ws(
    session: RealtimeSession,
    messages: list[dict],
    context: str,
    sio=None,
    drain_redis: bool = False,
) -> int:
    """Send pending text messages to the OpenAI sideband WebSocket.

    Serialized per session via session._flush_lock so concurrent HTTP routes
    cannot interleave ws.send writes or split a Redis LPOP drain across two
    response.create envelopes.

    For each message, extracts _turn_meta (if present) to create a
    VoiceTurn for turn tracking using the **frontend-generated** message
    IDs, persists the message pair to the DB, and emits chat:message:create
    so the frontend knows about the turn.  Strips _turn_meta only after
    successful send.  Sends a trailing response.create to trigger
    the model's reply.

    When *drain_redis* is True, also pops messages from the Redis pending
    store one at a time (via LPOP) after exhausting *messages*.  Messages
    stay in Redis until actually sent.

    Returns the number of messages successfully sent (0 if none).
    """
    if not session.ws:
        return 0
    if not messages and not drain_redis:
        return 0

    async with session._flush_lock:
        return await _flush_pending_to_ws_inner(session, messages, context, sio, drain_redis)


async def _flush_pending_to_ws_inner(  # noqa: C901 — two-phase drain (local list then Redis LPOP) sharing _send_one closure
    session: RealtimeSession,
    messages: list[dict],
    context: str,
    sio,
    drain_redis: bool,
) -> int:  # noqa: C901 — two-phase drain (local list then Redis LPOP) sharing _send_one closure; closure captures sent_count and session
    sent_count = 0

    async def _send_one(message: dict) -> bool:
        """Process and send a single pending message. Returns True on success."""
        turn_meta = message.get('_turn_meta')
        if turn_meta and session.turn_state:
            handoff = RealtimeTurnHandoff.from_legacy_dict(turn_meta)
            user_msg_id = handoff.user_msg_id or str(uuid4())
            asst_msg_id = handoff.asst_msg_id or str(uuid4())
            parent_id = handoff.tree_parent_id
            text_content = handoff.text_content
            item_id = handoff.item_id or message.get('item', {}).get('id', '')
            turn_files = handoff.files

            create_realtime_text_turn(
                session,
                item_id=item_id,
                user_message_id=user_msg_id,
                assistant_message_id=asst_msg_id,
                parent_message_id=parent_id,
                user_text=text_content,
            )

            try:
                await create_voice_turn_messages(
                    chat_id=session.chat_id,
                    parent_id=parent_id,
                    user_msg_id=user_msg_id,
                    asst_msg_id=asst_msg_id,
                    model_id=session.model_id,
                    user_content=text_content,
                    user_files=turn_files,
                )
                if sio:
                    await emit_to_user(
                        sio,
                        session.user_id,
                        session.chat_id,
                        asst_msg_id,
                        {
                            'type': 'chat:message:create',
                            'data': {
                                'chatId': session.chat_id,
                                'parentId': parent_id,
                                'userMessage': {
                                    'id': user_msg_id,
                                    'role': 'user',
                                    'content': text_content,
                                    **({'files': turn_files} if turn_files else {}),
                                },
                                'assistantMessage': {
                                    'id': asst_msg_id,
                                    'role': 'assistant',
                                    'content': '',
                                    'model': session.model_id,
                                },
                                'currentId': asst_msg_id,
                            },
                        },
                    )
            except Exception:
                log.exception('Failed to persist/emit pending text turn')

        ws = session.ws
        if not ws:
            return False

        # Strip _turn_meta only after confirming send
        message.pop('_turn_meta', None)
        event_id = str(uuid4())
        message['event_id'] = event_id
        session.track_event_id(event_id, f'{context}_flush')
        await ws.send(json.dumps(message))
        return True

    for message in messages:
        if await _send_one(message):
            sent_count += 1
        else:
            break

    if drain_redis and session.ws:
        while True:
            redis_msg = await pop_one_pending_text(session.session_id)
            if not redis_msg:
                break
            if await _send_one(redis_msg):
                sent_count += 1
            else:
                break

    if sent_count == 0:
        return 0

    ws = session.ws
    if not ws:
        session.pending_flush_response_create = True
        return sent_count
    if session.active_response_id:
        session.pending_flush_response_create = True
    else:
        response_event_id = str(uuid4())
        session.track_event_id(response_event_id, f'{context}_response_create')
        await ws.send(json.dumps({'type': 'response.create', 'event_id': response_event_id}))
    return sent_count


async def _maybe_apply_provider_idle_timeout(session: RealtimeSession) -> None:
    if not session.ws or session.config.vad_type != 'server_vad':
        return
    if not session.config.idle_timeout_ms:
        return

    try:
        idle_timeout_ms = int(session.config.idle_timeout_ms)
    except (TypeError, ValueError):
        return

    session_update = {
        'audio': {
            'input': {
                'turn_detection': {
                    'type': 'server_vad',
                    'threshold': session.config.server_vad_threshold,
                    'silence_duration_ms': session.config.server_vad_silence_duration_ms,
                    'prefix_padding_ms': session.config.server_vad_prefix_padding_ms,
                    'create_response': session.config.vad_create_response,
                    'interrupt_response': session.config.vad_interrupt_response,
                    'idle_timeout_ms': idle_timeout_ms,
                }
            }
        },
    }
    await session.send_event(
        'session.update',
        {'session': session_update},
        context='provider_idle_timeout_update',
    )


async def _execute_tool_calls(  # noqa: C901 — per-tool execute + emit + persist loop
    session: RealtimeSession, function_calls: list[dict], sio, turn: VoiceTurn | None = None
) -> set[str]:  # noqa: C901 — per-tool execute + emit + persist loop; iteration body is a single transactional unit
    """Execute tool calls. Returns the set of call_ids that were successfully resolved."""
    from open_webui.realtime.tool_runtime import execute_realtime_tool_call

    if not turn:
        turn = session.turn_state.get_active_turn()
    message_id = turn.assistant_message_id if turn else ''

    event_emitter = None
    if session.event_emitter_factory and message_id:
        event_emitter = await session.event_emitter_factory(
            {
                'user_id': session.user_id,
                'chat_id': session.chat_id,
                'message_id': message_id,
            }
        )

    event_caller = None
    if session.event_caller_factory and message_id:
        event_caller = await session.event_caller_factory(
            {
                'session_id': session.session_id,
                'chat_id': session.chat_id,
                'message_id': message_id,
            }
        )

    metadata = {
        'chat_id': session.chat_id,
        'message_id': message_id,
        'session_id': session.session_id,
    }

    pending_tool_cards: list[tuple[dict, dict]] = []
    resolved_call_ids: set[str] = set()

    if turn and turn.assistant_transcript and not turn.assistant_output:
        turn.assistant_output.append(
            build_assistant_output_message_item(text=turn.assistant_transcript, status='completed')
        )

    for function_call in function_calls:
        fn_name = function_call.get('name', '')
        fn_call_id = function_call.get('call_id', '') or function_call.get('id', '')

        tool_card = {
            'type': 'function_call',
            'id': fn_call_id,
            'call_id': fn_call_id,
            'name': fn_name,
            'arguments': function_call.get('arguments', '{}'),
            'status': 'in_progress',
        }
        pending_tool_cards.append((function_call, tool_card))
        if turn:
            turn.assistant_output.append(tool_card)

    if turn and pending_tool_cards:
        await emit_turn_output(session, turn, sio)

    for function_call, tool_card in pending_tool_cards:
        fn_name = function_call.get('name', '')
        fn_call_id = function_call.get('call_id', '') or function_call.get('id', '')
        if event_emitter:
            await event_emitter(
                {
                    'type': 'status',
                    'data': {
                        'description': f'Calling {fn_name}...',
                        'done': False,
                        'activity': 'tool',
                        'callId': session.call_id,
                    },
                }
            )

        result = await execute_realtime_tool_call(
            tool_call_id=fn_call_id,
            tool_function_name=fn_name,
            tool_args=function_call.get('arguments', '{}'),
            tools=session.tools_dict,
            request=session.tool_request,
            user=session.user,
            metadata=metadata,
            event_emitter=event_emitter,
            event_caller=event_caller,
        )

        if event_emitter:
            for source in result.get('sources', []):
                await event_emitter({'type': 'source', 'data': source})

        tool_card['status'] = 'failed' if result.get('failed') else 'completed'
        tool_card['arguments'] = function_call.get('arguments', '{}')

        if turn:
            result_item = {
                'type': 'function_call_output',
                'id': f'fco_{uuid4().hex}',
                'call_id': fn_call_id,
                'output': [{'type': 'input_text', 'text': result.get('content', '')}],
                'status': 'failed' if result.get('failed') else 'completed',
            }
            if result.get('files'):
                result_item['files'] = result.get('files')
            if result.get('embeds'):
                result_item['embeds'] = result.get('embeds')
            turn.assistant_output.append(result_item)

        if event_emitter:
            await event_emitter(
                {
                    'type': 'status',
                    'data': {
                        'description': (f'{fn_name} failed' if result.get('failed') else f'{fn_name} completed'),
                        'done': True,
                        'activity': 'tool',
                        'callId': session.call_id,
                    },
                }
            )

        await session.send_event(
            'conversation.item.create',
            {
                'item': {
                    'type': 'function_call_output',
                    'call_id': fn_call_id,
                    'output': result.get('content', ''),
                }
            },
            context=f'tool_result:{fn_name}',
        )
        resolved_call_ids.add(fn_call_id)

    if turn and turn.assistant_output:
        ensure_trailing_assistant_output_message(turn)
        await emit_turn_output(session, turn, sio)
        await persist_turn_output(session, turn)

    return resolved_call_ids


async def _persist_and_gc_turn(session: RealtimeSession, turn: VoiceTurn) -> None:
    """Persist turn output then GC the turn. GC only after persist completes."""
    await persist_turn_output(session, turn)
    maybe_gc_turn(session, turn)


async def _fire_deferred_response_create(session: RealtimeSession) -> bool:
    """Fire pending deferred response.create if any. Returns True if fired."""
    if session.pending_tool_response_create:
        session.pending_tool_response_create = False
        session.pending_flush_response_create = False
        await session.send_event('response.create', context='deferred_tool_response')
        return True
    elif session.pending_flush_response_create:
        session.pending_flush_response_create = False
        await session.send_event('response.create', context='deferred_flush_response')
        return True
    return False


async def _run_tool_calls_background(  # noqa: C901 — tool-batch success/failure feeding response.create scheduling
    session: RealtimeSession, function_calls: list[dict], sio, turn: VoiceTurn | None = None
) -> None:  # noqa: C901 — tool-batch success/failure decision tree feeding back into response.create scheduling
    """Execute tool calls in the background, then send response.create when safe."""
    tool_success = False
    resolved_call_ids: set[str] = set()
    try:
        resolved_call_ids = await _execute_tool_calls(session, function_calls, sio, turn=turn)
        tool_success = True
    except Exception:
        log.exception('Tool execution failed for session %s', session.session_id)
        # Send synthetic error outputs only for unresolved calls — OpenAI requires
        # function_call_output for every function_call before a new response can proceed.
        if session.state not in ('ending', 'idle'):
            for fc in function_calls:
                call_id = fc.get('call_id', '') or fc.get('id', '')
                if call_id and call_id not in resolved_call_ids:
                    await session.send_event(
                        'conversation.item.create',
                        {
                            'item': {
                                'type': 'function_call_output',
                                'call_id': call_id,
                                'output': 'Tool execution failed.',
                            }
                        },
                        context='tool_error_output',
                    )
        # Add error output items and update tool cards so the frontend
        # spinner flips to done (serialize_output checks for matching
        # function_call_output items, not the status field).
        if turn:
            error_outputs = []
            for item in turn.assistant_output:
                if item.get('type') == 'function_call' and item.get('status') == 'in_progress':
                    call_id = item.get('call_id', '')
                    if call_id and call_id not in resolved_call_ids:
                        item['status'] = 'failed'
                        error_outputs.append(
                            {
                                'type': 'function_call_output',
                                'call_id': call_id,
                                'output': [{'type': 'input_text', 'text': 'Tool execution failed.'}],
                                'status': 'failed',
                            }
                        )
            turn.assistant_output.extend(error_outputs)
            await emit_turn_output(session, turn, sio)
    finally:
        session.tool_executing = False

    if session.state in ('ending', 'idle'):
        return

    if session.active_response_id:
        session.pending_tool_response_create = True
    else:
        context = 'tool_response_create' if tool_success else 'tool_error_recovery'
        await session.send_event('response.create', context=context)


_SidebandEventHandler = Callable[[RealtimeSession, dict, Any], Awaitable[None]]
_HandlerT = TypeVar('_HandlerT', bound=_SidebandEventHandler)
_EVENT_HANDLERS: dict[str, _SidebandEventHandler] = {}


def _register_event(event_type: str) -> Callable[[_HandlerT], _HandlerT]:
    """Register an async handler for a single OpenAI Realtime WS event type.

    The handler receives (session, data, sio) — the same arguments
    _handle_sideband_event passes after its initial touch_activity() call.
    Returning from the handler is equivalent to falling through the
    original elif branch (or returning early from it).
    """
    def decorator(fn: _HandlerT) -> _HandlerT:
        if event_type in _EVENT_HANDLERS:
            raise ValueError(f'duplicate sideband event handler: {event_type}')
        _EVENT_HANDLERS[event_type] = fn
        return fn
    return decorator


@_register_event('session.created')
async def _on_session_created(session: RealtimeSession, data: dict, sio) -> None:
    session.session_created = True
    await maybe_emit_ready(session, sio)


@_register_event('session.updated')
async def _on_session_updated(session: RealtimeSession, data: dict, sio) -> None:
    session.session_update_ack = True
    await maybe_emit_ready(session, sio)


@_register_event('input_audio_buffer.speech_started')
async def _on_speech_started(session: RealtimeSession, data: dict, sio) -> None:
    # Order matters: timer rearm must reflect user input, not the new turn.
    item_id = data.get('item_id', '')
    active = session.turn_state.active_turn_id
    active_turn = session.turn_state.turns.get(active) if active else None
    log.debug(
        'realtime.speech_started item_id=%s assistant_responding=%s active_turn=%s asst_transcript_len=%d',
        item_id, session.assistant_responding, active,
        len(active_turn.assistant_transcript) if active_turn else 0,
    )
    await record_user_activity(session, sio)
    turn = await ensure_realtime_voice_turn(session, sio, item_id=item_id)
    if not turn:
        return


@_register_event('input_audio_buffer.speech_stopped')
async def _on_speech_stopped(session: RealtimeSession, data: dict, sio) -> None:
    await emit_status(sio, session.user_id, session.chat_id, session.call_id, 'Processing...', False)
    await restart_all_timers(session, sio)


@_register_event('input_audio_buffer.committed')
async def _on_audio_committed(session: RealtimeSession, data: dict, sio) -> None:
    # Order matters: timer rearm must reflect user input, not the new turn.
    await record_user_activity(session, sio)
    item_id = data.get('item_id', '')
    turn = await ensure_realtime_voice_turn(session, sio, item_id=item_id)
    if not turn:
        return


@_register_event('conversation.item.input_audio_transcription.delta')
async def _on_input_transcription_delta(session: RealtimeSession, data: dict, sio) -> None:
    delta = data.get('delta', '')
    item_id = data.get('item_id', '')
    turn = session.turn_state.get_turn_by_input_item(item_id)
    if turn:
        session.turn_state.append_user_transcript(turn.turn_id, delta)
        await emit_to_user(
            sio,
            session.user_id,
            session.chat_id,
            turn.user_message_id,
            {
                'type': 'chat:message:delta',
                'data': {'id': turn.user_message_id, 'content': delta},
            },
        )


@_register_event('conversation.item.input_audio_transcription.completed')
async def _on_input_transcription_completed(session: RealtimeSession, data: dict, sio) -> None:
    transcript = data.get('transcript', '')
    item_id = data.get('item_id', '')
    turn = session.turn_state.get_turn_by_input_item(item_id)
    if turn:
        session.turn_state.finalize_user_transcript(turn.turn_id, transcript, item_id=item_id)
        merged_transcript = turn.user_transcript

        if not transcript.strip() and len(turn.input_item_ids) > len(turn.finalized_item_ids):
            return

        if turn.is_empty and len(turn.input_item_ids) <= 1:
            await remove_orphan_turn_messages(
                session.chat_id,
                turn.user_message_id,
                turn.assistant_message_id,
                turn.parent_message_id,
            )
            await emit_to_user(
                sio,
                session.user_id,
                session.chat_id,
                turn.user_message_id,
                {'type': 'chat:message:prune'},
            )
            session.turn_state.gc_turn(turn.turn_id)
            return

        await emit_to_user(
            sio,
            session.user_id,
            session.chat_id,
            turn.user_message_id,
            {
                'type': 'replace',
                'data': {'id': turn.user_message_id, 'content': merged_transcript},
            },
        )

        async with chat_write_lock(session.chat_id):
            await _upsert_message_no_advance(
                session.chat_id,
                turn.user_message_id,
                {
                    'id': turn.user_message_id,
                    'role': 'user',
                    'content': merged_transcript,
                    'timestamp': int(time.time()),
                },
            )

        if turn.is_assistant_done:
            output_message = ensure_trailing_assistant_output_message(turn, status='completed')
            output_message['content'][-1]['text'] = turn.assistant_transcript
            await emit_turn_output(session, turn, sio, done=True)

        maybe_gc_turn(session, turn)


@_register_event('response.created')
async def _on_response_created(session: RealtimeSession, data: dict, sio) -> None:
    response = data.get('response', {})
    response_id = response.get('id', '')
    session.assistant_responding = True
    if response_id:
        session.active_response_id = response_id
        await sio.emit(
            'realtime:response_state',
            {
                'callId': session.call_id,
                'voiceSessionId': session.voice_session_id,
                'generation': session.generation,
                'responseId': response_id,
                'phase': 'started',
            },
            room=session.session_id,
        )

    if session.pending_transient_response_kind == 'idle_checkin' and response_id:
        session.transient_response_ids.add(response_id)
        session.pending_transient_response_kind = None
    else:
        turn = session.turn_state.get_pending_assistant_turn()
        if turn and response_id:
            session.turn_state.bind_response(response_id, turn.turn_id)

    if session.await_bootstrap_replay_response:
        session.await_bootstrap_replay_response = False
        await maybe_emit_ready(session, sio)


@_register_event('response.output_audio_transcript.delta')
async def _on_output_transcript_delta(session: RealtimeSession, data: dict, sio) -> None:
    delta = data.get('delta', '')
    response_id = data.get('response_id', '')
    if response_id in session.transient_response_ids:
        return
    turn = session.turn_state.get_turn_by_response(response_id)
    if turn:
        session.turn_state.append_assistant_transcript(turn.turn_id, delta)
        output_message = ensure_trailing_assistant_output_message(turn)
        output_message['content'][-1]['text'] = turn.assistant_transcript
        await emit_turn_output(session, turn, sio)


@_register_event('response.output_audio_transcript.done')
async def _on_output_transcript_done(session: RealtimeSession, data: dict, sio) -> None:
    transcript = data.get('transcript', '')
    response_id = data.get('response_id', '')
    if response_id in session.transient_response_ids:
        return
    turn = session.turn_state.get_turn_by_response(response_id)
    if turn:
        session.turn_state.finalize_assistant_transcript(turn.turn_id, transcript)
        output_message = ensure_trailing_assistant_output_message(turn, status='completed')
        output_message['content'][-1]['text'] = turn.assistant_transcript
        await emit_turn_output(session, turn, sio, done=turn.is_user_done)


@_register_event('output_audio_buffer.started')
async def _on_output_audio_buffer_started(session: RealtimeSession, data: dict, sio) -> None:
    response_id = data.get('response_id', '')
    if response_id in session.transient_response_ids:
        await cancel_checkin_timer(session)
    else:
        await cancel_all_timers(session)


@_register_event('output_audio_buffer.stopped')
async def _on_output_audio_buffer_stopped(session: RealtimeSession, data: dict, sio) -> None:
    response_id = data.get('response_id', '')
    is_checkin = response_id in session.transient_response_ids
    if is_checkin:
        session.transient_response_ids.discard(response_id)
        await restart_checkin_timer_only(session, sio)
    else:
        await restart_all_timers(session, sio)


@_register_event('response.done')
async def _on_response_done(session: RealtimeSession, data: dict, sio) -> None:  # noqa: C901 — response.done is the per-turn settling point: state cleanup, settings flush, function-call dispatch, persist, deferred replay, drain. Each sub-step is short; splitting further would scatter the ordering invariants across helpers.
    response = data.get('response', {})
    response_id = response.get('id', '')
    is_transient_response = response_id in session.transient_response_ids
    session.assistant_responding = False
    done_turn = session.turn_state.get_turn_by_response(response_id) if response_id else None
    log.debug(
        'realtime.response_done response_id=%s status=%s output_count=%d turn_id=%s user_msg=%s asst_len=%d transient=%s',  # noqa: E501
        response_id, response.get('status'), len(response.get('output') or []),
        done_turn.turn_id if done_turn else None,
        done_turn.user_message_id if done_turn else None,
        len(done_turn.assistant_transcript) if done_turn else 0,
        is_transient_response,
    )

    # Capture user-cancel state BEFORE cancel_requested_response_id is cleared
    was_user_cancel = (
        response.get('status') == 'cancelled'
        and response_id
        and session.cancel_requested_response_id == response_id
    )

    if response_id:
        if session.active_response_id == response_id:
            session.active_response_id = ''
        if session.cancel_requested_response_id == response_id:
            session.cancel_requested_response_id = ''
        await sio.emit(
            'realtime:response_state',
            {
                'callId': session.call_id,
                'voiceSessionId': session.voice_session_id,
                'generation': session.generation,
                'responseId': response_id,
                'phase': 'done',
            },
            room=session.session_id,
        )

    if is_transient_response:
        deferred = await _fire_deferred_response_create(session)
        if not deferred:
            await emit_status(sio, session.user_id, session.chat_id, session.call_id, '', True)
        return

    turn = session.turn_state.get_turn_by_response(response_id)

    if turn and len(turn.input_item_ids) > 1 and not turn.has_any_output():
        log.debug(
            'voice_turn.continuation_response_done_skip turn_id=%s response_id=%s input_items=%s',
            turn.turn_id, response_id, turn.input_item_ids,
        )
        session.turn_state.unbind_response(response_id)
        return

    if session.pending_settings_updates and session.ws:
        merged = {}
        for update in session.pending_settings_updates:
            setting = update.get('setting', '')
            value = update.get('value')
            if setting and value is not None:
                merged[setting] = value
        session.pending_settings_updates.clear()

        if merged:
            recreate_settings = {}
            for key in ('voice',):
                if key in merged:
                    recreate_settings[key] = merged.pop(key)

            if merged:
                session_update: dict = {}
                for key, val in merged.items():
                    if key == 'speed':
                        session_update.setdefault('audio', {}).setdefault('output', {})['speed'] = float(val)
                    elif key in ('max_output_tokens', 'max_tokens'):
                        session_update['max_output_tokens'] = (
                            int(val) if str(val).strip().lower() not in {'inf', '-1', ''} else 'inf'
                        )
                    elif key == 'noise_reduction':
                        session_update.setdefault('audio', {}).setdefault('input', {})['noise_reduction'] = (
                            {'type': val} if val else None
                        )
                    elif key == 'vad_type':
                        turn_detection = None
                        if val == 'semantic_vad':
                            turn_detection = {'type': 'semantic_vad'}
                        elif val == 'server_vad':
                            turn_detection = {'type': 'server_vad'}
                        session_update.setdefault('audio', {}).setdefault('input', {})['turn_detection'] = (
                            turn_detection
                        )

                ws = session.ws
                if ws:
                    event_id = str(uuid4())
                    session.track_event_id(event_id, f'settings_flush:{",".join(merged.keys())}')
                    session_update.setdefault('type', 'realtime')
                    await ws.send(
                        json.dumps(
                            {
                                'type': 'session.update',
                                'event_id': event_id,
                                'session': session_update,
                            }
                        )
                    )
            if recreate_settings:
                for key, value in recreate_settings.items():
                    await session.apply_setting(key, value, sio=sio)
                if session.state == 'reconnecting':
                    return

    function_calls = [item for item in response.get('output', []) if item.get('type') == 'function_call']
    if function_calls and session.tools_dict:
        session.tool_executing = True
        fire_and_forget(_run_tool_calls_background(session, function_calls, sio, turn=turn))
        return

    if turn:
        turn.is_assistant_done = True

        output_message = ensure_trailing_assistant_output_message(turn, status='completed')
        output_message['content'][-1]['text'] = turn.assistant_transcript

        await emit_turn_output(session, turn, sio, done=True)
        await _persist_and_gc_turn(session, turn)

    # Clear deferred flags on user-initiated cancellation only.
    # Server-side cancels (VAD interrupts) should still fire deferred.
    if was_user_cancel:
        session.pending_tool_response_create = False
        session.pending_flush_response_create = False

    # Fire deferred response.create AFTER settings/turn cleanup, BEFORE drain
    deferred = await _fire_deferred_response_create(session)
    if deferred:
        return

    await emit_status(sio, session.user_id, session.chat_id, session.call_id, '', True)

    await _flush_pending_to_ws(session, [], 'active_pending', sio=sio, drain_redis=True)


@_register_event('error')
async def _on_error(session: RealtimeSession, data: dict, sio) -> None:  # noqa: C901 — error event has three response paths: idle-timeout reject, duplicate-cancel race silencing, and the normal log+emit+maybe-end_session flow. Splitting would scatter the context-tag parsing across helpers.
    error = data.get('error', {})
    error_msg = error.get('message', str(error))
    event_id = data.get('event_id', '')

    context = session.lookup_event_id(event_id) if event_id else None
    if context == 'provider_idle_timeout_update':
        log.warning(
            'Provider idle timeout update rejected for %s: %s',
            session.session_id,
            error_msg,
        )
        session.config.idle_timeout_ms = ''
        return
    error_type = error.get('type', '')
    cancel_context_response_id = ''
    if isinstance(context, str) and context.startswith('response_cancel'):
        _, _, cancel_context_response_id = context.partition(':')
    is_duplicate_cancel_race = (
        isinstance(context, str)
        and context.startswith('response_cancel')
        and error_type == 'invalid_request_error'
        and 'no active response found' in error_msg.lower()
    )
    if is_duplicate_cancel_race:
        resolved_response_id = (
            cancel_context_response_id or session.cancel_requested_response_id or session.active_response_id
        )
        log.info(
            'Ignoring duplicate realtime cancel for %s: %s',
            session.session_id,
            error_msg,
        )
        if resolved_response_id and session.active_response_id == resolved_response_id:
            session.active_response_id = ''
        if resolved_response_id and session.cancel_requested_response_id == resolved_response_id:
            session.cancel_requested_response_id = ''
        if resolved_response_id:
            await sio.emit(
                'realtime:response_state',
                {
                    'callId': session.call_id,
                    'voiceSessionId': session.voice_session_id,
                    'generation': session.generation,
                    'responseId': resolved_response_id,
                    'phase': 'done',
                },
                room=session.session_id,
            )
        await _fire_deferred_response_create(session)
        return

    if context:
        log.error(
            'Sideband error for %s (action=%s, event_id=%s): %s',
            session.session_id,
            context,
            event_id,
            error_msg,
        )
    else:
        log.error('Sideband error for %s: %s', session.session_id, error_msg)

    await emit_to_user(
        sio,
        session.user_id,
        session.chat_id,
        '',
        {'type': 'chat:message:error', 'data': {'error': {'content': error_msg}}},
    )

    if error_type in (
        'rate_limit_exceeded',
        'server_error',
        'invalid_request_error',
    ):
        log.warning("Fatal error '%s' — ending session %s", error_type, session.session_id)
        await end_session(
            session,
            sio,
            'sideband_error',
            message=error_msg,
        )


async def _handle_sideband_event(
    session: RealtimeSession,
    event_type: str,
    data: dict,
    sio,
) -> None:
    """Route a single sideband event through the session's state machine."""
    session.touch_activity()

    handler = _EVENT_HANDLERS.get(event_type)
    if handler is not None:
        await handler(session, data, sio)


async def start_sideband(  # noqa: C901 — connection bootstrap pipeline: WS connect -> session.update -> bootstrap inject -> ready emit
    session: RealtimeSession,
    api_key: str,
    base_url: str,
    sio,
) -> None:  # noqa: C901 — connection bootstrap pipeline: WS connect -> session.update -> bootstrap injection -> ready emit
    """Connect the sideband WebSocket and start the event loop."""
    reset_bootstrap_flags(session)

    ws_base = base_url.replace('https://', 'wss://').replace('http://', 'ws://')
    url = f'{ws_base}/realtime?call_id={session.call_id}'

    import websockets

    try:
        session.ws = await websockets.connect(
            url,
            additional_headers={'Authorization': f'Bearer {api_key}'},
        )
        session.sideband_connected = True
        await emit_status(sio, session.user_id, session.chat_id, session.call_id, 'Preparing voice session...', False)

        replay_unanswered_user_turn = None
        if session.config.context_enabled:
            from open_webui.realtime.context_builder import (
                build_bootstrap_context,
                build_conversation_items,
                build_summary_conversation_item,
                generate_context_summary,
            )

            unanswered_last_user_turn = (
                session.config.context_unanswered_last_user_turn if not session.pending_messages else 'discard'
            )
            await emit_status(
                sio, session.user_id, session.chat_id, session.call_id, 'Loading conversation history...', False
            )
            bootstrap = await build_bootstrap_context(
                session.chat_id,
                recent_exchanges_limit=session.config.context_recent_exchanges_limit,
                older_summary_exchanges_limit=session.config.context_max_history_exchanges,
                older_summary_bytes_limit=session.config.context_max_history_bytes,
                unanswered_last_user_turn=unanswered_last_user_turn,
            )

            system_summary = None
            if session.config.context_summarize and bootstrap.summary_messages:
                await emit_status(
                    sio, session.user_id, session.chat_id, session.call_id, 'Summarizing conversation context...', False
                )
                from open_webui.realtime.tool_runtime import RealtimeToolRequest

                system_summary = await generate_context_summary(
                    bootstrap.summary_messages,
                    summary_prompt=session.config.context_summary_prompt,
                    summary_max_size=session.config.context_summary_max_size,
                    model_id=session.model_id,
                    chat_id=session.chat_id,
                    request=RealtimeToolRequest(session.app),
                    user=session.user,
                )

            items = []
            summary_item = build_summary_conversation_item(system_summary)
            if summary_item:
                items.append(summary_item)

            items.extend(build_conversation_items(bootstrap.replay_messages))

            replay_unanswered_user_turn = bootstrap.unresolved_user_turn

            # Resolve / create the assistant reply placeholder BEFORE we inject
            # context into the OpenAI conversation. If resolution fails (chat
            # row missing, ambiguous siblings) we still want the model to see
            # the prior chat history, but the unanswered user turn must NOT be
            # replayed — otherwise the model would respond to it without any
            # backing assistant message in our DB.
            assistant_message_id = None
            if replay_unanswered_user_turn and replay_unanswered_user_turn.get('id'):
                assistant_message_id = await resolve_realtime_voice_assistant_reply_message(
                    session.chat_id,
                    replay_unanswered_user_turn['id'],
                    session.model_id,
                    assistant_content=ASSISTANT_LISTENING_PLACEHOLDER,
                )
                if assistant_message_id:
                    items.extend(build_conversation_items([replay_unanswered_user_turn]))
                else:
                    log.warning(
                        'Skipping replay of unanswered user turn — resolve failed session=%s chat=%s',
                        session.session_id,
                        session.chat_id,
                    )
                    replay_unanswered_user_turn = None

            if items:
                await emit_status(sio, session.user_id, session.chat_id, session.call_id, 'Injecting context...', False)
            for item in items:
                event_id = str(uuid4())
                item_copy = {**item, 'event_id': event_id}
                session.track_event_id(event_id, 'context_injection')
                await session.ws.send(json.dumps(item_copy))

        if replay_unanswered_user_turn and replay_unanswered_user_turn.get('id'):
            if assistant_message_id:
                turn_id = str(uuid4())
                session.turn_state.create_turn(
                    turn_id=turn_id,
                    user_message_id=replay_unanswered_user_turn['id'],
                    assistant_message_id=assistant_message_id,
                    parent_message_id=replay_unanswered_user_turn.get('parentId') or '',
                )
                session.turn_state.finalize_user_transcript(
                    turn_id,
                    replay_unanswered_user_turn.get('content', ''),
                )
                await emit_to_user(
                    sio,
                    session.user_id,
                    session.chat_id,
                    assistant_message_id,
                    {
                        'type': 'chat:message:create',
                        'data': {
                            'chatId': session.chat_id,
                            'parentId': replay_unanswered_user_turn['id'],
                            'assistantMessage': {
                                'id': assistant_message_id,
                                'role': 'assistant',
                                'content': ASSISTANT_LISTENING_PLACEHOLDER,
                                'model': session.model_id,
                            },
                            'currentId': assistant_message_id,
                        },
                    },
                )
            else:
                log.warning(
                    'Unable to create assistant reply placeholder for replayed turn session=%s chat=%s',
                    session.session_id,
                    session.chat_id,
                )
                replay_unanswered_user_turn = None

        session.context_injected = True
        await emit_status(sio, session.user_id, session.chat_id, session.call_id, '', True)
        if replay_unanswered_user_turn:
            session.ready_auto_unmute = False
            session.await_bootstrap_replay_response = True
            replay_event_id = str(uuid4())
            session.track_event_id(replay_event_id, 'bootstrap_replay_response_create')
            await session.ws.send(
                json.dumps(
                    {
                        'type': 'response.create',
                        'event_id': replay_event_id,
                    }
                )
            )
        await maybe_emit_ready(session, sio)

        local_pending = list(session.pending_messages)
        sent = await _flush_pending_to_ws(session, local_pending, 'pending', sio=sio, drain_redis=True)
        if sent >= len(local_pending):
            session.pending_messages.clear()
        elif sent > 0:
            del session.pending_messages[:sent]

        session.pending_messages_flushed = True
        await maybe_emit_ready(session, sio)
        await _maybe_apply_provider_idle_timeout(session)

        # Re-drain once more after emitting ready — catches messages that
        # arrived between the first drain and the state transition.
        await _flush_pending_to_ws(session, [], 'late_pending', sio=sio, drain_redis=True)

        session.state = 'active'

        async for message in session.ws:
            try:
                event_data = json.loads(message)
            except (json.JSONDecodeError, TypeError):
                continue

            event_type = event_data.get('type', '')
            await _handle_sideband_event(session, event_type, event_data, sio)

        raise _SidebandClosed(getattr(session.ws, 'close_code', 1000) or 1000)
    except asyncio.CancelledError:
        pass
    finally:
        # Always clear bootstrap status so the mic is never stuck locked.
        try:
            await emit_status(sio, session.user_id, session.chat_id, session.call_id, '', True)
        except Exception:
            pass
        if session.ws:
            try:
                await session.ws.close()
            except Exception:
                pass
        session.ws = None
        session.sideband_connected = False


def _extract_error_code(exc: Exception) -> int:
    """Extract HTTP or WS close code from a websockets exception."""
    response = getattr(exc, 'response', None)
    if response is not None:
        return getattr(response, 'status_code', 0)
    status_code = getattr(exc, 'status_code', 0)
    if status_code:
        return status_code
    rcvd = getattr(exc, 'rcvd', None)
    if rcvd is not None:
        return getattr(rcvd, 'code', 0)
    return getattr(exc, 'code', 0)


async def start_sideband_with_retry(  # noqa: C901 — reconnect loop with exponential backoff + state recovery
    session: RealtimeSession,
    api_key: str,
    base_url: str,
    sio,
    max_retries: int = 2,
) -> None:  # noqa: C901 — reconnect loop with exponential backoff + state recovery; the conditionals encode retry policy
    """Start the sideband with exponential backoff reconnection."""
    for attempt in range(max_retries):
        if session.state in ('ending', 'idle'):
            return

        try:
            await start_sideband(session, api_key, base_url, sio)
            return
        except asyncio.CancelledError:
            return
        except Exception as exc:
            code = _extract_error_code(exc)

            if code in (401, 403):
                log.error(
                    'Sideband auth failed (%d) for session %s — not retrying',
                    code,
                    session.session_id,
                )
                await end_session(session, sio, 'auth_error', message=str(exc))
                return

            if code == 404:
                log.error(
                    'Sideband call_id invalid (404) for session %s — not retrying',
                    session.session_id,
                )
                await end_session(session, sio, 'sideband_error', message=str(exc))
                return

            if code in (1000, 1001):
                await end_session(session, sio, 'sideband_closed')
                return

            if attempt == 0:
                await sio.emit(
                    'realtime:sideband_lost',
                    {
                        'callId': session.call_id,
                        'attempt': 1,
                        'max_retries': max_retries,
                    },
                    room=session.session_id,
                )

            if code == 429:
                log.warning('Sideband rate-limited (429) for session %s', session.session_id)
            else:
                log.warning(
                    'Sideband attempt %d/%d failed for session %s (code=%d): %s',
                    attempt + 1,
                    max_retries,
                    session.session_id,
                    code,
                    exc,
                )

            if session.state not in ('ending', 'idle'):
                session.state = 'connecting'

        if attempt < max_retries - 1:
            await asyncio.sleep(2**attempt)

    log.error('All %d sideband retries failed for session %s', max_retries, session.session_id)
    await end_session(
        session,
        sio,
        'sideband_error',
        message='Open WebUI: Realtime sideband connection failed after retries.',
    )
