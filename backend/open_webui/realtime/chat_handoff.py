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
from open_webui.utils.files import get_image_url_from_base64
from open_webui.utils.misc import openai_chat_completion_message_template
from open_webui.realtime.pending_store import notify_pending_text, push_pending_text
from open_webui.utils.misc import get_last_user_message_item

log = logging.getLogger(__name__)


def _extract_image_files(content: Any, request: Any, chat_id: str, user: Any) -> list[dict]:
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
                    stored_url = get_image_url_from_base64(request, url, {'chat_id': chat_id}, user)
                    if stored_url:
                        file_id = stored_url.split('/files/')[-1].split('/')[0]
                        content_type = url.split(';')[0].split(':')[1] if ';' in url else 'image/png'
                        files.append({'type': 'image', 'url': file_id, 'content_type': content_type})
                except Exception:
                    log.exception('Failed to upload realtime image to storage')
    return files


def should_route_chat_to_realtime(request: Any, model: dict) -> bool:
    return model_uses_realtime(request, model) and not getattr(request.state, 'direct', False)


async def route_chat_completion_to_realtime(request: Any, form_data: dict, user: Any):
    metadata = form_data.get('metadata', {})
    message_item = get_last_user_message_item(form_data.get('messages', []))
    content = message_item.get('content', '') if message_item else ''

    session_id = metadata.get('session_id', '')
    chat_id = metadata.get('chat_id', '')
    model_id = form_data.get('model', '')

    if not session_id or not chat_id:
        raise Exception('No active realtime session.')

    image_files = _extract_image_files(content, request, chat_id, user) if isinstance(content, list) else []

    content_items = build_realtime_content_items(content)
    item_id = uuid4().hex
    text_content = extract_visible_text_from_content(content)

    parent_message = metadata.get('parent_message') or {}
    parent_message_id = metadata.get('parent_message_id', '')

    pending_msg = {
        'type': 'conversation.item.create',
        'item': {
            'id': item_id,
            'type': 'message',
            'role': 'user',
            'content': content_items,
        },
        '_turn_meta': {
            'item_id': item_id,
            'text_content': text_content,
            'chat_id': chat_id,
            'model_id': model_id,
            'user_id': user.id,
            # Frontend-generated IDs — the sideband MUST use these so the
            # frontend and backend agree on which message to update.
            'message_id': metadata.get('message_id', ''),
            'parent_message_id': parent_message_id,
            'parent_id': parent_message.get('parentId', ''),
            'files': image_files,
        },
    }

    await push_pending_text(session_id, pending_msg)
    await notify_pending_text(session_id)

    return openai_chat_completion_message_template(model_id, message='')
