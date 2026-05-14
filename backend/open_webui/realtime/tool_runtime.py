"""Realtime-owned tool resolution and execution runtime."""

import ast
import asyncio
import copy
import json
import logging
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

log = logging.getLogger(__name__)


async def execute_tool_call(
    tool_call_id: str,
    tool_function_name: str,
    tool_args: str,
    tools: dict,
    request,
    user,
    metadata: dict,
    messages: list,
    files: list,
    event_emitter,
    event_caller,
    citations_enabled: bool = True,
) -> dict:
    """Execute a single tool call and return the result.

    Parses arguments, invokes the tool (direct or callable),
    processes the result, and extracts citation sources.
    """
    from open_webui.utils.middleware import (
        get_citation_source_from_tool_result,
        process_tool_result,
        terminal_event_handler,
    )
    from open_webui.utils.tools import get_updated_tool_function

    tool_function_params = {}
    if tool_args and tool_args.strip():
        try:
            tool_function_params = ast.literal_eval(tool_args)
        except Exception as e:
            log.debug(e)
            try:
                tool_function_params = json.loads(tool_args)
            except Exception as e:
                log.error(f'Error parsing tool call arguments: {tool_args}')
                return {
                    'tool_call_id': tool_call_id,
                    'content': (
                        'Error: Tool call arguments could not be parsed. '
                        f'The model generated malformed or incomplete JSON for `{tool_function_name}`. '
                        'Please try again.'
                    ),
                }

    log.debug(f'Parsed args from {tool_args} to {tool_function_params}')

    tool_result = None
    tool = None
    tool_type = None
    direct_tool = False

    if tool_function_name in tools:
        tool = tools[tool_function_name]
        spec = tool.get('spec', {})

        tool_type = tool.get('type', '')
        direct_tool = tool.get('direct', False)

        try:
            allowed_params = spec.get('parameters', {}).get('properties', {}).keys()
            tool_function_params = {k: v for k, v in tool_function_params.items() if k in allowed_params}

            if direct_tool:
                tool_result = await event_caller(
                    {
                        'type': 'execute:tool',
                        'data': {
                            'id': str(uuid4()),
                            'name': tool_function_name,
                            'params': tool_function_params,
                            'server': tool.get('server', {}),
                            'session_id': metadata.get('session_id', None),
                        },
                    }
                )
            else:
                tool_function = await get_updated_tool_function(
                    function=tool['callable'],
                    extra_params={
                        '__messages__': messages,
                        '__files__': files,
                    },
                )
                tool_result = await tool_function(**tool_function_params)

        except Exception as e:
            tool_result = str(e)

    tool_result, tool_result_files, tool_result_embeds = await process_tool_result(
        request,
        tool_function_name,
        tool_result,
        tool_type,
        direct_tool,
        metadata,
        user,
    )

    await terminal_event_handler(
        tool_function_name,
        tool_function_params,
        tool_result,
        event_emitter,
    )

    sources = []
    if (
        citations_enabled
        and tool_function_name
        in [
            'search_web',
            'fetch_url',
            'view_file',
            'view_knowledge_file',
            'query_knowledge_files',
        ]
        and tool_result
    ):
        try:
            sources = get_citation_source_from_tool_result(
                tool_name=tool_function_name,
                tool_params=tool_function_params,
                tool_result=tool_result,
                tool_id=tool.get('tool_id', '') if tool else '',
            )
        except Exception as e:
            log.exception(f'Error extracting citation source: {e}')

    return {
        'tool_call_id': tool_call_id,
        'content': str(tool_result) if tool_result else '',
        **({'files': tool_result_files} if tool_result_files else {}),
        **({'embeds': tool_result_embeds} if tool_result_embeds else {}),
        'sources': sources,
        'params': tool_function_params,
    }


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
        metadata: dict | None = None,
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
    tool_ids: list[str] | None,
    tool_servers: list[dict] | None,
    terminal_id: str | None,
    features: dict | None,
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


def _normalize_tool_args(tool_args: Any) -> str:
    if isinstance(tool_args, dict):
        return json.dumps(tool_args)
    if isinstance(tool_args, str):
        return tool_args
    return '{}'


def _resolve_event_handlers(event_caller, event_emitter):
    """Provide safe defaults for event_caller / event_emitter in sideband context."""

    async def _no_caller(payload):
        # Raise so the caller's try/except marks the tool result as failed.
        # Returning a JSON-encoded error here would be propagated as successful
        # tool content.
        raise RuntimeError('Error: realtime tool requires a browser session but none is attached.')

    async def _noop_emitter(x):
        pass

    return (event_caller or _no_caller, event_emitter or _noop_emitter)


async def _emit_result_artifacts(event_emitter, result: dict) -> None:
    """Emit files/embeds events that the underlying inline path does not emit."""
    if not event_emitter:
        return
    if result.get('files'):
        await event_emitter({'type': 'files', 'data': {'files': result['files']}})
    if result.get('embeds'):
        await event_emitter({'type': 'embeds', 'data': {'embeds': result['embeds']}})


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
    """Thin sideband wrapper around execute_tool_call.

    Adds realtime-specific adaptations before and after the underlying call:
    - Pre-parse dict args to JSON string (defensive)
    - Guard event_caller for direct tools without browser session
    - Guard event_emitter (may be None in sideband context)
    - Track tool_failed status in the returned dict
    - Emit files/embeds events (the underlying inline path does not emit these)
    """
    safe_caller, safe_emitter = _resolve_event_handlers(event_caller, event_emitter)

    result = await execute_tool_call(
        tool_call_id=tool_call_id,
        tool_function_name=tool_function_name,
        tool_args=_normalize_tool_args(tool_args),
        tools=tools,
        request=request,
        user=user,
        metadata=metadata,
        messages=[],
        files=[],
        event_emitter=safe_emitter,
        event_caller=safe_caller,
        citations_enabled=True,
    )

    content = result.get('content', '')
    result['failed'] = content.startswith('Error:') or tool_function_name not in tools

    await _emit_result_artifacts(event_emitter, result)

    return result
