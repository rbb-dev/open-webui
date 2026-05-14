"""Thin payload enrichment for realtime text turns.

When voice mode is active for a model, the chat completion path
short-circuits into route_chat_completion_to_realtime and never runs
process_chat_payload.  That means typed-during-voice turns lose
pipeline inlet filters, memory injection, web-search retrieval, and
model-knowledge retrieval — features the user has explicitly enabled
silently stop working.

This module runs only the voice-relevant subset of that pipeline.
Tool resolution / MCP / native-FC orchestration are deliberately
skipped — realtime tools are negotiated at session-mint time and the
sideband worker holds them for the lifetime of the call.

Output: enriched form_data plus a context_text string the handoff
prepends as a transient input_text item before the user's content.
The transient block decays with the turn, which matches realtime
conversation semantics — we never mutate session.instructions on a
per-turn basis.

DESIGN TRADEOFF — handler reuse vs shared helper in middleware:
The cleaner long-term factoring would be a process_chat_payload_minimal
helper inside middleware.py that both the chat-completion and realtime
paths share, so feature drift cannot occur.  This module reuses the
individual handlers instead (process_pipeline_inlet_filter,
chat_memory_handler, chat_web_search_handler, chat_completion_files_handler,
apply_system_prompt_to_body) without refactoring middleware.  An
upstream PR may revisit this tradeoff; for now the realtime fork
isolates its changes to its own subpackage.
"""

import copy
import logging
from typing import Any

log = logging.getLogger(__name__)


def _compute_system_delta(before: str, after: str) -> str | None:
    """Return the text enrichment appended to the system message, or
    None if enrichment did not append (no change, or a wholesale
    rewrite we cannot safely merge with realtime's baked instructions).

    Trailing whitespace is normalized before the prefix check so an
    inlet that strips/adds a trailing newline does not flip a benign
    append into a discarded rewrite.
    """
    before_norm = (before or '').rstrip()
    after_norm = (after or '').rstrip()
    if after_norm == before_norm:
        return None
    if after_norm.startswith(before_norm):
        return after_norm[len(before_norm):].strip() or None
    log.warning('realtime.enrich: inlet rewrote system message; system delta discarded')
    return None


def _collect_knowledge_files(model_knowledge: Any) -> list[dict]:
    """Mirror process_chat_payload's knowledge → file-ref expansion.

    Mirrors the same edge cases as middleware.py (including the
    falsy-string fallthrough for empty collection_name values).
    """
    files: list[dict] = []
    if not model_knowledge:
        return files
    for item in model_knowledge:
        if item.get('collection_name'):
            files.append({
                'id': item.get('collection_name'),
                'name': item.get('name'),
                'legacy': True,
            })
        elif item.get('collection_names'):
            files.append({
                'name': item.get('name'),
                'type': 'collection',
                'collection_names': item.get('collection_names'),
                'legacy': True,
            })
        else:
            files.append(item)
    return files


def _resolve_native_fc(metadata: dict, model: dict) -> bool:
    """Source-of-truth: model.info.params.function_calling.

    Stock middleware reads from metadata.params, but that dict is
    populated by apply_params_to_form_data(form_data, model) at the
    start of process_chat_payload — which the realtime handoff
    bypasses.  Reading directly from the model record also avoids a
    client-spoofable bypass via metadata.params.function_calling.
    """
    model_params = model.get('info', {}).get('params', {}) or {}
    if model_params.get('function_calling') == 'native':
        return True
    return (metadata.get('params') or {}).get('function_calling') == 'native'


async def _apply_folder_system_prompt(
    request: Any,
    form_data: dict,
    user: Any,
    metadata: dict,
) -> dict:
    """Re-apply the chat's folder-level system prompt for this turn.

    Folder prompts are applied at negotiate-time (session mint) but
    can contain time-sensitive template variables (e.g.
    {{CURRENT_DATETIME}}) that need to refresh each turn, and a chat
    may have been moved between folders since session start.
    """
    chat_id = metadata.get('chat_id') or ''
    if not chat_id or chat_id.startswith('local:') or chat_id.startswith('channel:'):
        return form_data
    if not user:
        return form_data
    from open_webui.models.chats import Chats
    from open_webui.models.folders import Folders
    from open_webui.utils.payload import apply_system_prompt_to_body

    user_id = getattr(user, 'id', None)
    if not user_id:
        return form_data

    try:
        folder_id = await Chats.get_chat_folder_id(chat_id, user_id)
    except Exception:
        log.exception('realtime.enrich: folder id lookup failed')
        return form_data
    if not folder_id:
        return form_data
    try:
        folder = await Folders.get_folder_by_id_and_user_id(folder_id, user_id)
    except Exception:
        log.exception('realtime.enrich: folder fetch failed')
        return form_data
    if not folder or not folder.data or 'system_prompt' not in folder.data:
        return form_data
    try:
        return await apply_system_prompt_to_body(
            folder.data['system_prompt'], form_data, metadata, user,
        )
    except Exception:
        log.exception('realtime.enrich: folder system prompt apply failed')
        return form_data


async def enrich_realtime_text_turn(  # noqa: C901 — sequential pipeline mirroring process_chat_payload's voice-relevant subset; each step is short and decomposing would require passing 6+ locals through helper signatures
    request: Any,
    form_data: dict,
    user: Any,
    metadata: dict,
    model: dict,
) -> tuple[dict, str]:
    """Run inlets + memory/web_search/knowledge handlers for a typed
    realtime turn.  Resolves any retrieved files into a <source>
    context block and returns the additional context as a string the
    caller prepends to the realtime user content.

    Returns (form_data, context_text).

    All handlers are wrapped so an enrichment failure never breaks
    the handoff — the user just loses that one feature for this turn.

    LIMITATION: image parts added by inlet filters are not picked up
    by the caller's pre-inlet _extract_image_files() pass.  Inlets
    that need to add image inputs are not currently supported in
    voice mode.
    """
    from open_webui.models.functions import Functions
    from open_webui.models.users import UserModel
    from open_webui.routers.pipelines import process_pipeline_inlet_filter
    from open_webui.socket.main import get_event_call, get_event_emitter
    from open_webui.utils.filter import (
        get_sorted_filter_ids,
        process_filter_functions,
    )
    from open_webui.utils.middleware import (
        chat_completion_files_handler,
        chat_memory_handler,
        chat_web_search_handler,
        get_source_context,
        get_system_oauth_token,
        load_messages_from_db,
    )
    from open_webui.utils.misc import (
        get_last_user_message,
        get_system_message,
    )
    from open_webui.utils.task import rag_template

    models = request.app.state.MODELS

    if not form_data.get('messages'):
        chat_id = metadata.get('chat_id')
        user_message_id = metadata.get('user_message_id')
        if chat_id and user_message_id and not chat_id.startswith('local:') and not chat_id.startswith('channel:'):
            db_messages = await load_messages_from_db(chat_id, user_message_id)
            if db_messages:
                form_data['messages'] = db_messages
    form_data.setdefault('messages', [])

    form_data = await _apply_folder_system_prompt(request, form_data, user, metadata)

    original_system_msg = get_system_message(form_data.get('messages', []))
    original_system_text = (original_system_msg or {}).get('content', '') if original_system_msg else ''

    event_emitter = await get_event_emitter(metadata)
    event_caller = await get_event_call(metadata)
    try:
        oauth_token = await get_system_oauth_token(request, user)
    except Exception:
        oauth_token = None
    extra_params = {
        '__event_emitter__': event_emitter,
        '__event_call__': event_caller,
        '__user__': user.model_dump() if isinstance(user, UserModel) else {},
        '__metadata__': metadata,
        '__oauth_token__': oauth_token,
        '__request__': request,
        '__model__': model,
        '__chat_id__': metadata.get('chat_id'),
        '__message_id__': metadata.get('message_id'),
    }

    try:
        form_data = await process_pipeline_inlet_filter(request, form_data, user, models)
    except Exception:
        log.exception('realtime.enrich: pipeline inlet filter failed')

    try:
        filter_ids = await get_sorted_filter_ids(request, model, metadata.get('filter_ids', []))
        filter_functions = await Functions.get_functions_by_ids(filter_ids)
        form_data, _flags = await process_filter_functions(
            request=request,
            filter_functions=filter_functions,
            filter_type='inlet',
            form_data=form_data,
            extra_params=extra_params,
        )
    except Exception:
        log.exception('realtime.enrich: inlet filter functions failed')

    features = form_data.pop('features', None) or {}
    native_fc = _resolve_native_fc(metadata, model)

    if features.get('memory') and not native_fc:
        try:
            form_data = await chat_memory_handler(request, form_data, extra_params, user)
        except Exception:
            log.exception('realtime.enrich: memory handler failed')

    if features.get('web_search') and not native_fc:
        try:
            form_data = await chat_web_search_handler(request, form_data, extra_params, user)
        except Exception:
            log.exception('realtime.enrich: web_search handler failed')

    model_knowledge = model.get('info', {}).get('meta', {}).get('knowledge', False)
    if model_knowledge and not native_fc:
        knowledge_files = _collect_knowledge_files(model_knowledge)
        if knowledge_files:
            files = list(form_data.get('files') or [])
            files.extend(knowledge_files)
            form_data['files'] = files

    sources: list[dict] = []
    pending_files = form_data.pop('files', None)
    if pending_files:
        # Detach so chat_completion_files_handler can't mutate the
        # caller's messages list through aliasing.  We discard the
        # returned body — only sources are kept.
        body = copy.copy(form_data)
        body['messages'] = list(form_data.get('messages', []))
        # pending_files was just popped from form_data['files'], so any
        # pre-existing metadata.files would be a stale/duplicate copy.
        # Overwriting is intentional — chat_completion_files_handler reads
        # files from body['metadata']['files'].
        body['metadata'] = {**(body.get('metadata') or {}), 'files': pending_files}
        try:
            _body, flags = await chat_completion_files_handler(request, body, extra_params, user)
            sources = (flags or {}).get('sources') or []
        except Exception:
            log.exception('realtime.enrich: files handler failed')

    new_system_msg = get_system_message(form_data.get('messages', []))
    new_system_text = (new_system_msg or {}).get('content', '') if new_system_msg else ''
    system_delta = _compute_system_delta(original_system_text, new_system_text)

    context_parts: list[str] = []
    if system_delta:
        context_parts.append(system_delta)
    if sources:
        rendered = get_source_context(sources).strip()
        if rendered:
            try:
                user_message_text = get_last_user_message(form_data.get('messages', [])) or ''
                rag_rendered = await rag_template(
                    request.app.state.config.RAG_TEMPLATE,
                    rendered,
                    user_message_text,
                )
                context_parts.append(rag_rendered.strip())
            except Exception:
                log.exception('realtime.enrich: RAG template render failed; using raw <source> block')
                context_parts.append(rendered)

    context_text = '\n\n'.join(part for part in context_parts if part).strip()

    log.info(
        'realtime.enrich complete chat_id=%s memory=%s web_search=%s '
        'knowledge=%s sources=%d sys_delta_chars=%d context_chars=%d',
        metadata.get('chat_id'),
        bool(features.get('memory') and not native_fc),
        bool(features.get('web_search') and not native_fc),
        bool(model_knowledge and not native_fc),
        len(sources),
        len(system_delta or ''),
        len(context_text),
    )

    return form_data, context_text
