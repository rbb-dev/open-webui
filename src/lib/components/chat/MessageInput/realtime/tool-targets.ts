export function getRealtimeToolTargets(
	selectedToolIds: string[],
	toolServers: Array<any>,
	terminalServers: Array<any>
) {
	const toolIds: string[] = [];
	const toolServerIds: Array<string | number> = [];

	for (const toolId of selectedToolIds) {
		if (toolId.startsWith('direct_server:')) {
			const serverId = toolId.replace('direct_server:', '');
			const parsedServerId = parseInt(serverId);
			toolServerIds.push(Number.isNaN(parsedServerId) ? serverId : parsedServerId);
		} else {
			toolIds.push(toolId);
		}
	}

	// Bundle anonymous (no-id) terminals because they live in user settings and
	// the backend has no way to look them up otherwise. Admin-configured
	// terminals (with id) are intentionally NOT bundled here — they're resolved
	// server-side via the realtime payload's terminalId field.
	return {
		toolIds,
		toolServers: [
			...(toolServers ?? []).filter(
				(server, idx) => toolServerIds.includes(idx) || toolServerIds.includes(server?.id)
			),
			...(terminalServers ?? []).filter((terminal) => !terminal.id)
		]
	};
}
