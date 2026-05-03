"""Realtime-owned tool resolution and execution runtime."""

import asyncio
import copy
import json
import logging
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Optional

log = logging.getLogger(__name__)


async def cleanup_realtime_mcp_clients(mcp_clients: dict) -> None:
    """Disconnect all MCP clients in reverse order. Idempotent and safe.

    MUST run in the SAME asyncio task that opened the connections.  The MCP
    SDK's streamable_http transport uses anyio TaskGroups whose cancel scope
    must exit in the same task it entered.  Per upstream MCPClient.disconnect()
    guidance, do NOT wrap calls in asyncio.shield() or asyncio.wait_for() —
    both create a new task and trigger the cross-task cancel-scope error this
    cleanup is meant to prevent.
    """
    if not mcp_clients:
        return

    for client in reversed(list(mcp_clients.values())):
        try:
            await client.disconnect()
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            log.debug('Realtime MCP disconnect error: %s', exc)


@dataclass
class RealtimeToolAuthContext:
    session_token: str = ''
    oauth_session_id: str = ''


class RealtimeToolRequest:
    """Minimal request-like object for realtime tool resolution/execution.

    This intentionally carries only explicit server-owned auth state needed by
    OWUI tool handlers instead of replaying the original browser request.
    """

    def __init__(
        self,
        app: Any,
        *,
        session_token: str = '',
        oauth_session_id: str = '',
        metadata: Optional[dict] = None,
    ):
        self.app = app
        self.state = SimpleNamespace(
            token=SimpleNamespace(credentials=session_token),
            direct=False,
            metadata=metadata or {},
        )
        self.cookies = {'oauth_session_id': oauth_session_id} if oauth_session_id else {}
        self.headers = {}
        self.query_params = {}
        self.method = 'POST'
        self.url = SimpleNamespace(path='')
        self.path_params = {}
        self.scope = {}

    def __getattr__(self, name):
        raise AttributeError(f"{self.__class__.__name__} has no attribute '{name}'")


async def resolve_realtime_tools(
    *,
    app: Any,
    user: Any,
    model: dict,
    chat_id: str,
    session_id: str,
    tool_ids: Optional[list[str]],
    tool_servers: Optional[list[dict]],
    terminal_id: Optional[str],
    features: Optional[dict],
    auth_context: RealtimeToolAuthContext,
    event_emitter: Any = None,
    event_caller: Any = None,
) -> tuple[RealtimeToolRequest, dict[str, dict], dict]:
    # Import lazily to avoid a module cycle:
    # socket.main -> session_service -> tool_runtime -> middleware -> socket.main
    from open_webui.utils.middleware import process_chat_payload

    metadata = {
        'chat_id': chat_id,
        'message_id': '',
        'session_id': session_id,
        'user_id': getattr(user, 'id', ''),
        'tool_ids': tool_ids or [],
        'tool_servers': copy.deepcopy(tool_servers or []),
        'terminal_id': terminal_id,
        'features': features or {},
        'params': {'function_calling': 'native'},
    }
    request = RealtimeToolRequest(
        app,
        session_token=auth_context.session_token,
        oauth_session_id=auth_context.oauth_session_id,
        metadata=metadata,
    )
    request.state.metadata = metadata

    tool_form = {
        'model': model.get('id', ''),
        'messages': [],
        'tool_ids': tool_ids or [],
        'tool_servers': copy.deepcopy(tool_servers or []),
        'terminal_id': terminal_id,
        'features': features or {},
    }

    try:
        tool_form, metadata, _ = await process_chat_payload(
            request,
            tool_form,
            user,
            metadata,
            model,
        )
    except Exception:
        # Drain any partially-connected MCP clients in this same task before
        # propagating the error upwards.  Skipping this would let the anyio
        # cancel scope leak across tasks during GC and surface as an unrelated
        # RuntimeError later.
        await cleanup_realtime_mcp_clients(metadata.get('mcp_clients') or {})
        raise

    return request, metadata.get('tools', {}), metadata.get('mcp_clients') or {}


async def execute_realtime_tool_call(
    *,
    tool_call_id: str,
    tool_function_name: str,
    tool_args,
    tools: dict,
    request,
    user,
    metadata: dict,
    event_emitter=None,
    event_caller=None,
) -> dict:
    """Thin sideband wrapper around middleware.execute_tool_call.

    Adds realtime-specific adaptations before and after the stock call:
    - Pre-parse dict args to JSON string (defensive)
    - Guard event_caller for direct tools without browser session
    - Guard event_emitter (may be None in sideband context)
    - Track tool_failed status in the returned dict
    - Emit files/embeds events (stock inline path does not emit these)
    """
    from open_webui.utils.middleware import execute_tool_call

    if isinstance(tool_args, dict):
        tool_args = json.dumps(tool_args)
    elif not isinstance(tool_args, str):
        tool_args = '{}'

    actual_event_caller = event_caller
    if event_caller is None:

        async def _no_caller(payload):
            return json.dumps({'error': 'No browser session for direct tool'})

        actual_event_caller = _no_caller

    # Async no-op emitter so stock execute_tool_call doesn't crash
    # on await event_emitter(...) inside terminal_event_handler
    async def _noop_emitter(x):
        pass

    actual_event_emitter = event_emitter or _noop_emitter

    result = await execute_tool_call(
        tool_call_id=tool_call_id,
        tool_function_name=tool_function_name,
        tool_args=tool_args,
        tools=tools,
        request=request,
        user=user,
        metadata=metadata,
        messages=[],
        files=[],
        event_emitter=actual_event_emitter,
        event_caller=actual_event_caller,
        citations_enabled=True,
    )

    failed = False
    content = result.get('content', '')
    if content.startswith('Error:'):
        failed = True
    if tool_function_name not in tools:
        failed = True

    result['failed'] = failed

    # Post-process: emit files/embeds events (stock path doesn't do this)
    if event_emitter:
        result_files = result.get('files')
        result_embeds = result.get('embeds')
        if result_files:
            await event_emitter({'type': 'files', 'data': {'files': result_files}})
        if result_embeds:
            await event_emitter({'type': 'embeds', 'data': {'embeds': result_embeds}})

    return result
