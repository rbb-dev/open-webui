<script lang="ts">
	import { toast } from 'svelte-sonner';
	import { onMount, getContext } from 'svelte';

	import { settings, config } from '$lib/stores';
	import { getRealtimeClientConfig } from '$lib/components/chat/MessageInput/realtime/api';

	import Switch from '$lib/components/common/Switch.svelte';

	const i18n = getContext('i18n');

	// --- Realtime voice state ---
	let showAdvancedRealtimeSettings = false;
	let realtimeAutoUnmuteWhenReady = false;
	let realtimeVoice = 'marin';
	let realtimeSpeed = 1;
	let realtimeVadType: 'semantic_vad' | 'server_vad' | 'push_to_talk' = 'server_vad';
	let realtimeSemanticVadEagerness: 'low' | 'medium' | 'high' | 'auto' = 'auto';
	let realtimeServerVadThreshold = 0.5;
	let realtimeServerVadSilenceDurationMs = 500;
	let realtimeServerVadPrefixPaddingMs = 300;
	let realtimeNoiseReduction: 'near_field' | 'far_field' | '' = 'near_field';
	let realtimeVadCreateResponse = true;
	let realtimeVadInterruptResponse = true;
	let realtimeClientConfig = null;
	let realtimeVoiceOptions: { id: string; name: string }[] = [];

	// --- Public API: expose current settings to parent ---
	export function getRealtimeSettings() {
		return {
			autoUnmuteWhenReady: realtimeAutoUnmuteWhenReady,
			voice: realtimeVoice,
			speed: realtimeSpeed,
			vadType: realtimeVadType,
			semanticVadEagerness: realtimeSemanticVadEagerness,
			serverVadThreshold: realtimeServerVadThreshold,
			serverVadSilenceDurationMs: realtimeServerVadSilenceDurationMs,
			serverVadPrefixPaddingMs: realtimeServerVadPrefixPaddingMs,
			noiseReduction: realtimeNoiseReduction,
			vadCreateResponse: realtimeVadCreateResponse,
			vadInterruptResponse: realtimeVadInterruptResponse
		};
	}

	// --- Lifecycle ---
	onMount(async () => {
		if ($config?.audio?.realtime?.enabled) {
			realtimeClientConfig = await getRealtimeClientConfig(localStorage.token).catch((e) => {
				toast.error(`${e}`);
				return null;
			});
			realtimeVoiceOptions = realtimeClientConfig?.capabilities?.voices ?? [];
		}

		const realtimeDefaults = realtimeClientConfig?.defaults ?? {};
		const realtimeSettings = $settings?.audio?.realtime ?? {};

		realtimeAutoUnmuteWhenReady = realtimeSettings.autoUnmuteWhenReady ?? false;
		realtimeVoice = realtimeSettings.voice ?? realtimeDefaults.voice ?? 'marin';
		realtimeSpeed = realtimeSettings.speed ?? realtimeDefaults.speed ?? 1;
		realtimeVadType = realtimeSettings.vadType ?? realtimeDefaults.vad_type ?? 'server_vad';
		realtimeSemanticVadEagerness =
			realtimeSettings.semanticVadEagerness ?? realtimeDefaults.semantic_vad_eagerness ?? 'auto';
		realtimeServerVadThreshold =
			realtimeSettings.serverVadThreshold ?? realtimeDefaults.server_vad_threshold ?? 0.5;
		realtimeServerVadSilenceDurationMs =
			realtimeSettings.serverVadSilenceDurationMs ??
			realtimeDefaults.server_vad_silence_duration_ms ??
			500;
		realtimeServerVadPrefixPaddingMs =
			realtimeSettings.serverVadPrefixPaddingMs ??
			realtimeDefaults.server_vad_prefix_padding_ms ??
			300;
		realtimeNoiseReduction =
			realtimeSettings.noiseReduction ?? realtimeDefaults.noise_reduction ?? 'near_field';
		realtimeVadCreateResponse =
			realtimeSettings.vadCreateResponse ?? realtimeDefaults.vad_create_response ?? true;
		realtimeVadInterruptResponse =
			realtimeSettings.vadInterruptResponse ?? realtimeDefaults.vad_interrupt_response ?? true;
		if (realtimeVoice && !realtimeVoiceOptions.some((option) => option.id === realtimeVoice)) {
			realtimeVoiceOptions = [{ id: realtimeVoice, name: realtimeVoice }, ...realtimeVoiceOptions];
		}
	});
</script>

<hr class=" border-gray-100/30 dark:border-gray-850/30" />

<div class="space-y-3">
	<div>
		<div class=" mb-1 text-sm font-medium">
			{$i18n.t('settings.personal.audio.sections.realtimeVoice.title')}
		</div>
		<div class="text-xs text-gray-400 dark:text-gray-500">
			{$i18n.t('settings.personal.audio.sections.realtimeVoice.description')}
		</div>
	</div>

	<div>
		<div class="py-0.5 flex w-full justify-between">
			<div class="self-center text-xs font-medium">
				{$i18n.t('settings.personal.audio.realtime.autoUnmuteMicrophoneWhenCallIsReady.label')}
			</div>
			<div class="mt-1">
				<Switch bind:state={realtimeAutoUnmuteWhenReady} />
			</div>
		</div>
		<div class="text-xs text-gray-400 dark:text-gray-500">
			{$i18n.t('settings.personal.audio.realtime.autoUnmuteMicrophoneWhenCallIsReady.description')}
		</div>
	</div>

	<div>
		<div class=" py-0.5 flex w-full justify-between">
			<div class=" self-center text-xs font-medium">
				{$i18n.t('settings.personal.audio.realtime.voice.label')}
			</div>
			<div class="flex items-center relative">
				<select
					class="cursor-pointer w-fit pr-8 rounded-sm px-2 p-1 text-xs bg-transparent outline-hidden text-right"
					bind:value={realtimeVoice}
				>
					{#each realtimeVoiceOptions as realtimeVoiceOption}
						<option value={realtimeVoiceOption.id}>{realtimeVoiceOption.name}</option>
					{/each}
				</select>
			</div>
		</div>
		<div class="text-xs text-gray-400 dark:text-gray-500">
			{$i18n.t('settings.personal.audio.realtime.voice.description')}
		</div>
	</div>

	<div>
		<div class="py-0.5 flex w-full justify-between gap-3">
			<div class="self-center text-xs font-medium">
				{$i18n.t('settings.personal.audio.realtime.speechSpeed.label')}
			</div>
			<input
				type="number"
				step="0.05"
				min="0.25"
				max="1.5"
				class="w-24 rounded-lg py-1 px-2 text-right text-xs bg-gray-50 dark:text-gray-300 dark:bg-gray-850 outline-hidden"
				bind:value={realtimeSpeed}
			/>
		</div>
		<div class="text-xs text-gray-400 dark:text-gray-500">
			{$i18n.t('settings.personal.audio.realtime.speechSpeed.description')}
		</div>
	</div>

	<div>
		<div class=" py-0.5 flex w-full justify-between">
			<div class=" self-center text-xs font-medium">
				{$i18n.t('settings.personal.audio.realtime.voiceActivityDetectionVad.label')}
			</div>
			<div class="flex items-center relative">
				<select
					class="cursor-pointer w-fit pr-8 rounded-sm px-2 p-1 text-xs bg-transparent outline-hidden text-right"
					bind:value={realtimeVadType}
				>
					<option value="semantic_vad">{$i18n.t('Smart (AI-powered)')}</option>
					<option value="server_vad">{$i18n.t('Standard (volume-based)')}</option>
					<option value="push_to_talk">{$i18n.t('Push to Talk')}</option>
				</select>
			</div>
		</div>
		<div class="text-xs text-gray-400 dark:text-gray-500">
			{$i18n.t('settings.personal.audio.realtime.voiceActivityDetectionVad.description')}
		</div>
	</div>

	<div class="flex justify-between items-center text-sm">
		<div class="font-medium">
			{$i18n.t('settings.personal.audio.realtime.advancedSettings.label')}
		</div>
		<button
			class=" text-xs font-medium {($settings?.highContrastMode ?? false)
				? 'text-gray-800 dark:text-gray-100'
				: 'text-gray-400 dark:text-gray-500'}"
			type="button"
			aria-expanded={showAdvancedRealtimeSettings}
			on:click={() => {
				showAdvancedRealtimeSettings = !showAdvancedRealtimeSettings;
			}}
		>
			{showAdvancedRealtimeSettings ? $i18n.t('Hide') : $i18n.t('Show')}
		</button>
	</div>

	{#if showAdvancedRealtimeSettings}
		<div class="space-y-3">
			<div>
				<div class="py-0.5 flex w-full justify-between">
					<div class=" self-center text-xs font-medium">
						{$i18n.t('settings.personal.audio.realtime.noiseReduction.label')}
					</div>
					<div class="flex items-center relative">
						<select
							class="cursor-pointer w-fit pr-8 rounded-sm px-2 p-1 text-xs bg-transparent outline-hidden text-right"
							bind:value={realtimeNoiseReduction}
						>
							<option value="near_field">
								{$i18n.t('Close range (headset/laptop)')}
							</option>
							<option value="far_field">
								{$i18n.t('Far range (room mic/speaker)')}
							</option>
							<option value="">{$i18n.t('None')}</option>
						</select>
					</div>
				</div>
				<div class="text-xs text-gray-400 dark:text-gray-500">
					{$i18n.t('settings.personal.audio.realtime.noiseReduction.description')}
				</div>
			</div>

			<div class=" py-0.5 flex w-full justify-between">
				<div class=" self-center text-xs font-medium">
					{$i18n.t('settings.personal.audio.realtime.autoRespondAfterSpeech.label')}
				</div>
				<div class="mt-1">
					<Switch bind:state={realtimeVadCreateResponse} />
				</div>
			</div>
			<div class="text-xs text-gray-400 dark:text-gray-500">
				{$i18n.t('settings.personal.audio.realtime.autoRespondAfterSpeech.description')}
			</div>

			<div class=" py-0.5 flex w-full justify-between">
				<div class=" self-center text-xs font-medium">
					{$i18n.t('settings.personal.audio.realtime.allowSpeechInterruption.label')}
				</div>
				<div class="mt-1">
					<Switch bind:state={realtimeVadInterruptResponse} />
				</div>
			</div>
			<div class="text-xs text-gray-400 dark:text-gray-500">
				{$i18n.t('settings.personal.audio.realtime.allowSpeechInterruption.description')}
			</div>

			{#if realtimeVadType === 'semantic_vad'}
				<div>
					<div class=" py-0.5 flex w-full justify-between">
						<div class=" self-center text-xs font-medium">
							{$i18n.t('settings.personal.audio.realtime.responseEagerness.label')}
						</div>
						<div class="flex items-center relative">
							<select
								class="cursor-pointer w-fit pr-8 rounded-sm px-2 p-1 text-xs bg-transparent outline-hidden text-right"
								bind:value={realtimeSemanticVadEagerness}
							>
								<option value="auto">{$i18n.t('Auto')}</option>
								<option value="low">{$i18n.t('Low')}</option>
								<option value="medium">{$i18n.t('Medium')}</option>
								<option value="high">{$i18n.t('High')}</option>
							</select>
						</div>
					</div>
					<div class="text-xs text-gray-400 dark:text-gray-500">
						{$i18n.t('settings.personal.audio.realtime.responseEagerness.description')}
					</div>
				</div>
			{:else if realtimeVadType === 'server_vad'}
				<div class="space-y-3">
					<div>
						<div class="py-0.5 flex w-full justify-between gap-3">
							<div class="self-center text-xs font-medium">
								{$i18n.t('settings.personal.audio.realtime.volumeThreshold.label')}
							</div>
							<input
								type="number"
								step="0.1"
								min="0"
								max="1"
								class="w-24 rounded-lg py-1 px-2 text-right text-xs bg-gray-50 dark:text-gray-300 dark:bg-gray-850 outline-hidden"
								bind:value={realtimeServerVadThreshold}
							/>
						</div>
						<div class="text-xs text-gray-400 dark:text-gray-500">
							{$i18n.t('settings.personal.audio.realtime.volumeThreshold.description')}
						</div>
					</div>

					<div>
						<div class="py-0.5 flex w-full justify-between gap-3">
							<div class="self-center text-xs font-medium">
								{$i18n.t('settings.personal.audio.realtime.silenceDurationMs.label')}
							</div>
							<input
								type="number"
								min="0"
								class="w-24 rounded-lg py-1 px-2 text-right text-xs bg-gray-50 dark:text-gray-300 dark:bg-gray-850 outline-hidden"
								bind:value={realtimeServerVadSilenceDurationMs}
							/>
						</div>
						<div class="text-xs text-gray-400 dark:text-gray-500">
							{$i18n.t('settings.personal.audio.realtime.silenceDurationMs.description')}
						</div>
					</div>

					<div>
						<div class="py-0.5 flex w-full justify-between gap-3">
							<div class="self-center text-xs font-medium">
								{$i18n.t('settings.personal.audio.realtime.prefixPaddingMs.label')}
							</div>
							<input
								type="number"
								min="0"
								class="w-24 rounded-lg py-1 px-2 text-right text-xs bg-gray-50 dark:text-gray-300 dark:bg-gray-850 outline-hidden"
								bind:value={realtimeServerVadPrefixPaddingMs}
							/>
						</div>
						<div class="text-xs text-gray-400 dark:text-gray-500">
							{$i18n.t('settings.personal.audio.realtime.prefixPaddingMs.description')}
						</div>
					</div>
				</div>
			{/if}
		</div>
	{/if}
</div>
