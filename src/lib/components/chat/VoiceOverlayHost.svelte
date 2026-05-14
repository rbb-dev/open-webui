<script lang="ts">
	import { mobile, realtimeClientConfig, showCallOverlay } from '$lib/stores';

	import Drawer from '../common/Drawer.svelte';
	import ResizableSidePanel from '../common/ResizableSidePanel.svelte';
	import CallOverlay from './MessageInput/CallOverlay.svelte';
	import RealtimeOverlay from './MessageInput/RealtimeOverlay.svelte';
	import { modelHasRealtimeCapability } from './MessageInput/realtime/model-capabilities';

	export let models = [];
	export let modelId = '';
	export let chatId = null;
	export let selectedToolIds: string[] = [];
	export let toolServers = [];
	export let features = {};
	export let terminalId: string | null = null;
	export let systemPrompt = '';
	export let files = [];
	export let submitPrompt: Function = () => {};
	export let stopResponse: Function = () => {};
	export let eventTarget: EventTarget;

	let voiceOverlayWidth = 350;

	const useRealtimeOverlay = () =>
		$showCallOverlay && modelHasRealtimeCapability(models, modelId, $realtimeClientConfig);

	const closeOverlay = () => {
		showCallOverlay.set(false);
	};
</script>

{#if $mobile}
	{#if $showCallOverlay}
		<Drawer
			show={$showCallOverlay}
			onClose={closeOverlay}
			className="min-h-[100dvh] !bg-white dark:!bg-gray-850"
		>
			<div class="h-[100dvh] flex flex-col bg-white text-gray-700 dark:bg-black dark:text-gray-300">
				{#if useRealtimeOverlay()}
					<RealtimeOverlay
						bind:files
						{submitPrompt}
						{modelId}
						{chatId}
						{selectedToolIds}
						{toolServers}
						{features}
						{terminalId}
						{systemPrompt}
						on:close={closeOverlay}
					/>
				{:else}
					<CallOverlay
						bind:files
						{submitPrompt}
						{stopResponse}
						{modelId}
						{chatId}
						{eventTarget}
						on:close={closeOverlay}
					/>
				{/if}
			</div>
		</Drawer>
	{/if}
{:else if $showCallOverlay}
	<ResizableSidePanel
		open={$showCallOverlay}
		bind:width={voiceOverlayWidth}
		minWidth={350}
		minSiblingWidth={360}
		closeOnDragBelowMinWidth
		storageKey="chatVoiceOverlaySize"
		resizerId="voice-overlay-resizer"
		className="z-10 bg-white dark:bg-gray-850"
		onClose={closeOverlay}
	>
		<div
			class="w-full h-full flex justify-center bg-white text-gray-700 dark:bg-black dark:text-gray-300"
		>
			{#if useRealtimeOverlay()}
				<RealtimeOverlay
					bind:files
					{submitPrompt}
					{modelId}
					{chatId}
					{selectedToolIds}
					{toolServers}
					{features}
					{terminalId}
					{systemPrompt}
					on:close={closeOverlay}
				/>
			{:else}
				<CallOverlay
					bind:files
					{submitPrompt}
					{stopResponse}
					{modelId}
					{chatId}
					{eventTarget}
					on:close={closeOverlay}
				/>
			{/if}
		</div>
	</ResizableSidePanel>
{/if}
