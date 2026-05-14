"""Typed chat handoff into the realtime runtime.

In multi-worker mode, the HTTP POST lands on a random worker but the
realtime session lives on the Socket.IO-owning worker.  We push the
pending text message into a Redis-backed store (keyed by session_id)
and return immediately.  The sideband on the correct worker drains the
store during bootstrap and after each model response.
"""

import logging
from typing import Any
from uuid import uuid4

from open_webui.realtime.catalog import model_uses_realtime
from open_webui.realtime.chat_sync import (
    build_realtime_content_items,
    extract_visible_text_from_content,
)
from open_webui.realtime.contracts import RealtimeTurnHandoff
from open_webui.realtime.pending_store import notify_pending_text, push_pending_text
from open_webui.realtime.turn_enrichment import enrich_realtime_text_turn
from open_webui.utils.files import get_image_url_from_base64
from open_webui.utils.misc import (
    get_last_user_message_item,
    openai_chat_completion_message_template,
)

log = logging.getLogger(__name__)


async def _extract_image_files(content: Any, request: Any, chat_id: str, user: Any) -> list[dict]:
    """Extract image data URLs from message content, upload to OWUI storage, return file references."""
    if not isinstance(content, list):
        return []
    files = []
    for part in content:
        if isinstance(part, dict) and part.get('type') == 'image_url':
            image_url = part.get('image_url', {})
            url = image_url.get('url', '') if isinstance(image_url, dict) else str(image_url)
            if url.startswith('data:image/'):
                try:
                    stored_url = await get_image_url_from_base64(request, url, {'chat_id': chat_id}, user)
                    if stored_url:
                        file_id = stored_url.split('/files/')[-1].split('/')[0]
                        content_type = url.split(';')[0].split(':')[1] if ';' in url else 'image/png'
                        files.append({'type': 'image', 'url': file_id, 'content_type': content_type})
                except Exception:
                    log.exception('Failed to upload realtime image to storage')
    return files


def _dedupe_files(files: list[dict]) -> list[dict]:
    """Dedupe file refs by (type, url) so the same image referenced via both
    inline data-URL extraction and metadata.user_message.files doesn't get
    sent twice to the model."""
    seen: set[tuple[str, str]] = set()
    out: list[dict] = []
    for f in files:
        if not isinstance(f, dict):
            continue
        key = (str(f.get('type', '')), str(f.get('url', '')))
        if key in seen or not key[1]:
            continue
        seen.add(key)
        out.append(f)
    return out


async def _file_id_to_data_url(file_id: str, user_id: str) -> str | None:
    """Resolve a storage file_id to a base64 data URL the OpenAI Realtime API
    can consume directly.

    The realtime WS connection runs on OpenAI's network and cannot fetch from
    our private storage URLs, so we read the bytes locally and inline them as
    a data URL.
    """
    import asyncio
    import base64

    from open_webui.models.files import Files
    from open_webui.storage.provider import Storage

    try:
        file_obj = await Files.get_file_by_id_and_user_id(file_id, user_id)
        if not file_obj:
            return None
        local_path = await asyncio.to_thread(Storage.get_file, file_obj.path)
        with open(local_path, 'rb') as fh:
            data = fh.read()
        content_type = (file_obj.meta or {}).get('content_type') or 'image/png'
        return f'data:{content_type};base64,{base64.b64encode(data).decode("ascii")}'
    except Exception:
        log.exception('Failed to rehydrate storage file_id=%s for realtime handoff', file_id)
        return None


async def _content_items_with_storage_files(  # noqa: C901 — three image-source shapes (data URL in content, data URL in files, storage path/file_id) and one dedup pass; splitting would scatter the already_present set across helpers
    base_content: Any,
    files: list[dict],
    user_id: str,
) -> list[dict]:
    """Extend the realtime content_items list with storage-only image refs
    rehydrated as inline data URLs so the OpenAI Realtime API can see them.

    Skips entries whose url already appears as a data URL inside base_content
    (those were captured by _extract_image_files).
    """
    items = build_realtime_content_items(base_content)

    already_present: set[str] = set()
    for it in items:
        if it.get('type') == 'input_image':
            url = it.get('image_url', '')
            if isinstance(url, str) and url.startswith('data:image/'):
                already_present.add(url)

    for f in files:
        content_type = (f.get('content_type') or '')
        if f.get('type') != 'image' and not content_type.startswith('image/'):
            continue
        url_field = str(f.get('url', ''))
        if not url_field:
            continue

        # RealtimeOverlay.svelte uploads images via FileReader.readAsDataURL
        # and submits user_message.files=[{type:'image', url:'data:image/...'}]
        # — the URL is the data URL itself, not a storage path. Pass it through
        # to the model directly without trying to rehydrate from storage.
        if url_field.startswith('data:'):
            if url_field not in already_present:
                items.append({'type': 'input_image', 'image_url': url_field})
                already_present.add(url_field)
            continue

        # Storage-backed file: url_field is either a bare file_id (after
        # _extract_image_files post-processed an inline data URL) or a full
        # /api/v1/files/<id>/content path (legacy / other client paths).
        if '/files/' in url_field:
            file_id = url_field.split('/files/')[-1].split('/')[0]
        else:
            file_id = url_field
        data_url = await _file_id_to_data_url(file_id, user_id)
        if data_url and data_url not in already_present:
            items.append({'type': 'input_image', 'image_url': data_url})
            already_present.add(data_url)
        elif not data_url:
            log.warning('realtime.handoff: failed to rehydrate image file_id=%s url=%s', file_id, url_field)

    return items


def should_route_chat_to_realtime(request: Any, model: dict) -> bool:
    return model_uses_realtime(request, model) and not getattr(request.state, 'direct', False)


async def route_chat_completion_to_realtime(request: Any, form_data: dict, user: Any):
    metadata = form_data.get('metadata', {})
    message_item = get_last_user_message_item(form_data.get('messages', []))
    content = message_item.get('content', '') if message_item else ''

    session_id = metadata.get('session_id', '')
    chat_id = metadata.get('chat_id', '')
    model_id = form_data.get('model', '')

    if not session_id:
        raise Exception(
            'Realtime handoff: missing metadata.session_id — the client must include '
            'session_id (socket.id) when posting to a realtime-capable model.',
        )
    if not chat_id:
        raise Exception(
            'Realtime handoff: missing metadata.chat_id — cannot route the user turn '
            'to a realtime sideband without a chat context.',
        )

    user_message = metadata.get('user_message') or metadata.get('parent_message') or {}
    user_message_id = (
        metadata.get('user_message_id')
        or metadata.get('parent_message_id')
        or user_message.get('id', '')
    )

    # File sources to union, in priority order:
    #   1. inline data URLs in content (first-upload, pre-compression handoff)
    #   2. metadata.user_message.files (frontend's authoritative attached list)
    #   3. metadata.files (legacy / alternate field on some submission paths)
    inline_files = await _extract_image_files(content, request, chat_id, user) if isinstance(content, list) else []
    user_message_files = user_message.get('files') if isinstance(user_message.get('files'), list) else []
    metadata_files = metadata.get('files') if isinstance(metadata.get('files'), list) else []
    image_files = _dedupe_files([*inline_files, *user_message_files, *metadata_files])

    # Snapshot what the user actually typed BEFORE enrichment.  The
    # persisted chat record and chat:message:create frontend payload use
    # this text — they must reflect the user's words, not anything inlet
    # filters or retrieval added downstream.
    original_text = extract_visible_text_from_content(content)

    # Run the voice-relevant subset of the chat enrichment pipeline so a
    # typed-during-voice turn still benefits from inlet filters, memory,
    # web_search and knowledge retrieval.  Failures inside the helper are
    # already logged; the turn is still pushed without enrichment.
    context_text = ''
    model_obj = request.app.state.MODELS.get(model_id) if model_id else None
    if model_obj:
        try:
            form_data, context_text = await enrich_realtime_text_turn(
                request, form_data, user, metadata, model_obj,
            )
        except Exception:
            log.exception('realtime.handoff: enrichment failed; pushing turn without context')
            context_text = ''

        enriched_item = get_last_user_message_item(form_data.get('messages', []))
        if enriched_item is not None:
            enriched_content = enriched_item.get('content', content)
            if enriched_content:
                content = enriched_content
    else:
        log.debug('realtime.handoff: model not in MODELS registry, skipping enrichment model_id=%s', model_id)

    # Rehydrate storage-only file refs (file_id) into inline data URLs so the
    # OpenAI Realtime WS receives the bytes — it cannot fetch from our storage.
    content_items = await _content_items_with_storage_files(content, image_files, user.id)
    if context_text:
        # Transient per-turn context — auto-decays after the response, so we
        # never mutate session.instructions for per-turn retrieval results.
        content_items = [{'type': 'input_text', 'text': context_text}, *content_items]
    item_id = uuid4().hex
    text_content = original_text

    handoff = RealtimeTurnHandoff(
        item_id=item_id,
        text_content=text_content,
        chat_id=chat_id,
        model_id=model_id,
        user_id=user.id,
        user_msg_id=user_message_id,
        asst_msg_id=metadata.get('message_id', ''),
        tree_parent_id=user_message.get('parentId', ''),
        files=image_files or None,
    )

    pending_msg = {
        'type': 'conversation.item.create',
        'item': {
            'id': item_id,
            'type': 'message',
            'role': 'user',
            'content': content_items,
        },
        '_turn_meta': handoff.to_legacy_dict(),
    }

    await push_pending_text(session_id, pending_msg)
    await notify_pending_text(session_id)

    log.info(
        'realtime.handoff.push session_id=%s chat_id=%s item_id=%s has_files=%s',
        session_id,
        chat_id,
        item_id,
        bool(image_files),
    )

    return openai_chat_completion_message_template(model_id, message='')
