<script lang="ts">
	import { toast } from 'svelte-sonner';
	import { onMount, getContext } from 'svelte';

	import { getRealtimeClientConfig } from '$lib/components/chat/MessageInput/realtime/api';

	import SensitiveInput from '$lib/components/common/SensitiveInput.svelte';
	import Tooltip from '$lib/components/common/Tooltip.svelte';
	import Textarea from '$lib/components/common/Textarea.svelte';
	import Plus from '$lib/components/icons/Plus.svelte';
	import Minus from '$lib/components/icons/Minus.svelte';
	import Switch from '$lib/components/common/Switch.svelte';

	import type { Writable } from 'svelte/store';
	import type { i18n as i18nType } from 'i18next';

	const i18n = getContext<Writable<i18nType>>('i18n');

	export let initialRtConfig: any = null;

	// Realtime
	let RT_ENGINE = '';
	let RT_API_KEY = '';
	let RT_API_BASE_URL = '';
	let RT_MODELS: string[] = [];
	let RT_MODEL_INPUT = '';
	let RT_VOICE = '';
	let RT_VAD_TYPE = 'server_vad';
	let RT_SERVER_VAD_THRESHOLD = 0.5;
	let RT_SERVER_VAD_SILENCE_DURATION_MS = 500;
	let RT_SERVER_VAD_PREFIX_PADDING_MS = 300;
	let RT_SEMANTIC_VAD_EAGERNESS = 'auto';
	let RT_TRANSCRIPTION_MODEL = '';
	let RT_NOISE_REDUCTION = '';
	let RT_MAX_RESPONSE_OUTPUT_TOKENS = '';
	let RT_CONTEXT_ENABLED = false;
	let RT_CONTEXT_RECENT_EXCHANGES_LIMIT = 10;
	let RT_CONTEXT_MAX_HISTORY_EXCHANGES = 40;
	let RT_CONTEXT_MAX_HISTORY_BYTES = 16000;
	let RT_CONTEXT_SUMMARIZE = false;
	let RT_CONTEXT_UNANSWERED_LAST_USER_TURN = 'discard';
	let RT_CONTEXT_SUMMARY_PROMPT = '';
	let RT_CONTEXT_SUMMARY_MAX_SIZE = 2000;
	let RT_SPEED = 1.0;
	let RT_TRANSCRIPTION_PROMPT = '';
	let RT_VAD_IDLE_TIMEOUT_MS = '';
	let RT_VAD_CREATE_RESPONSE = true;
	let RT_VAD_INTERRUPT_RESPONSE = true;
	let RT_SESSION_TIMEOUT = 180;
	let RT_IDLE_CALL_CHECKIN_INTERVAL = 45;
	let RT_IDLE_CALL_CHECKIN_PROMPT =
		'System message: user is idle and needs a gentle push to talk to you, say something friendly to engage the user.';
	let RT_TRUNCATION_STRATEGY = 'auto';
	let RT_TRUNCATION_RETENTION_RATIO = 0.8;
	let RT_TRUNCATION_TOKEN_LIMIT = '';
	let RT_TASK_MODEL_WARNING = false;

	let realtimeVoiceOptions: { id: string; name: string }[] = [];

	export function getConfig() {
		return {
			ENGINE: RT_ENGINE,
			API_KEY: RT_API_KEY,
			API_BASE_URL: RT_API_BASE_URL,
			MODELS: RT_MODELS,
			VOICE: RT_VOICE,
			VAD_TYPE: RT_VAD_TYPE,
			SERVER_VAD_THRESHOLD: RT_SERVER_VAD_THRESHOLD,
			SERVER_VAD_SILENCE_DURATION_MS: RT_SERVER_VAD_SILENCE_DURATION_MS,
			SERVER_VAD_PREFIX_PADDING_MS: RT_SERVER_VAD_PREFIX_PADDING_MS,
			SEMANTIC_VAD_EAGERNESS: RT_SEMANTIC_VAD_EAGERNESS,
			TRANSCRIPTION_MODEL: RT_TRANSCRIPTION_MODEL,
			NOISE_REDUCTION: RT_NOISE_REDUCTION,
			MAX_RESPONSE_OUTPUT_TOKENS: RT_MAX_RESPONSE_OUTPUT_TOKENS,
			CONTEXT_ENABLED: RT_CONTEXT_ENABLED,
			CONTEXT_RECENT_EXCHANGES_LIMIT: RT_CONTEXT_RECENT_EXCHANGES_LIMIT,
			CONTEXT_MAX_HISTORY_EXCHANGES: RT_CONTEXT_MAX_HISTORY_EXCHANGES,
			CONTEXT_MAX_HISTORY_BYTES: RT_CONTEXT_MAX_HISTORY_BYTES,
			CONTEXT_SUMMARIZE: RT_CONTEXT_SUMMARIZE,
			CONTEXT_UNANSWERED_LAST_USER_TURN: RT_CONTEXT_UNANSWERED_LAST_USER_TURN,
			CONTEXT_SUMMARY_PROMPT: RT_CONTEXT_SUMMARY_PROMPT,
			CONTEXT_SUMMARY_MAX_SIZE: RT_CONTEXT_SUMMARY_MAX_SIZE,
			SPEED: RT_SPEED,
			TRANSCRIPTION_PROMPT: RT_TRANSCRIPTION_PROMPT,
			VAD_IDLE_TIMEOUT_MS: RT_VAD_IDLE_TIMEOUT_MS,
			VAD_CREATE_RESPONSE: RT_VAD_CREATE_RESPONSE,
			VAD_INTERRUPT_RESPONSE: RT_VAD_INTERRUPT_RESPONSE,
			SESSION_TIMEOUT: RT_SESSION_TIMEOUT,
			IDLE_CALL_CHECKIN_INTERVAL: RT_IDLE_CALL_CHECKIN_INTERVAL,
			IDLE_CALL_CHECKIN_PROMPT: RT_IDLE_CALL_CHECKIN_PROMPT,
			TRUNCATION_STRATEGY: RT_TRUNCATION_STRATEGY,
			TRUNCATION_RETENTION_RATIO: RT_TRUNCATION_RETENTION_RATIO,
			TRUNCATION_TOKEN_LIMIT: RT_TRUNCATION_TOKEN_LIMIT
		};
	}

	onMount(async () => {
		const realtimeClientConfig = await getRealtimeClientConfig(localStorage.token).catch((e) => {
			toast.error(`${e}`);
			return null;
		});
		realtimeVoiceOptions = realtimeClientConfig?.capabilities?.voices ?? [];

		if (initialRtConfig) {
			RT_ENGINE = initialRtConfig.ENGINE;
			RT_API_KEY = initialRtConfig.API_KEY;
			RT_API_BASE_URL = initialRtConfig.API_BASE_URL;
			RT_MODELS = initialRtConfig.MODELS ?? [];
			RT_VOICE = initialRtConfig.VOICE;
			RT_VAD_TYPE = initialRtConfig.VAD_TYPE;
			RT_SERVER_VAD_THRESHOLD = initialRtConfig.SERVER_VAD_THRESHOLD;
			RT_SERVER_VAD_SILENCE_DURATION_MS = initialRtConfig.SERVER_VAD_SILENCE_DURATION_MS;
			RT_SERVER_VAD_PREFIX_PADDING_MS = initialRtConfig.SERVER_VAD_PREFIX_PADDING_MS;
			RT_SEMANTIC_VAD_EAGERNESS = initialRtConfig.SEMANTIC_VAD_EAGERNESS;
			RT_TRANSCRIPTION_MODEL = initialRtConfig.TRANSCRIPTION_MODEL;
			RT_NOISE_REDUCTION = initialRtConfig.NOISE_REDUCTION;
			RT_MAX_RESPONSE_OUTPUT_TOKENS = initialRtConfig.MAX_RESPONSE_OUTPUT_TOKENS;
			RT_CONTEXT_ENABLED = initialRtConfig.CONTEXT_ENABLED ?? false;
			RT_CONTEXT_RECENT_EXCHANGES_LIMIT = initialRtConfig.CONTEXT_RECENT_EXCHANGES_LIMIT ?? 10;
			RT_CONTEXT_MAX_HISTORY_EXCHANGES = initialRtConfig.CONTEXT_MAX_HISTORY_EXCHANGES ?? 40;
			RT_CONTEXT_MAX_HISTORY_BYTES = initialRtConfig.CONTEXT_MAX_HISTORY_BYTES ?? 16000;
			RT_CONTEXT_SUMMARIZE = initialRtConfig.CONTEXT_SUMMARIZE ?? false;
			RT_CONTEXT_UNANSWERED_LAST_USER_TURN =
				initialRtConfig.CONTEXT_UNANSWERED_LAST_USER_TURN ?? 'discard';
			RT_CONTEXT_SUMMARY_PROMPT = initialRtConfig.CONTEXT_SUMMARY_PROMPT ?? '';
			RT_CONTEXT_SUMMARY_MAX_SIZE = initialRtConfig.CONTEXT_SUMMARY_MAX_SIZE ?? 2000;
			RT_SPEED = initialRtConfig.SPEED ?? 1.0;
			RT_TRANSCRIPTION_PROMPT = initialRtConfig.TRANSCRIPTION_PROMPT ?? '';
			RT_VAD_IDLE_TIMEOUT_MS = initialRtConfig.VAD_IDLE_TIMEOUT_MS ?? '';
			RT_VAD_CREATE_RESPONSE = initialRtConfig.VAD_CREATE_RESPONSE ?? true;
			RT_VAD_INTERRUPT_RESPONSE = initialRtConfig.VAD_INTERRUPT_RESPONSE ?? true;
			RT_SESSION_TIMEOUT = initialRtConfig.SESSION_TIMEOUT ?? 180;
			RT_IDLE_CALL_CHECKIN_INTERVAL = initialRtConfig.IDLE_CALL_CHECKIN_INTERVAL ?? 45;
			RT_IDLE_CALL_CHECKIN_PROMPT =
				initialRtConfig.IDLE_CALL_CHECKIN_PROMPT ??
				'System message: user is idle and needs a gentle push to talk to you, say something friendly to engage the user.';
			RT_TRUNCATION_STRATEGY = initialRtConfig.TRUNCATION_STRATEGY ?? 'auto';
			RT_TRUNCATION_RETENTION_RATIO = initialRtConfig.TRUNCATION_RETENTION_RATIO ?? 0.8;
			RT_TRUNCATION_TOKEN_LIMIT = initialRtConfig.TRUNCATION_TOKEN_LIMIT ?? '';
			RT_TASK_MODEL_WARNING = initialRtConfig.TASK_MODEL_WARNING ?? false;
			if (RT_VOICE && !realtimeVoiceOptions.some((option) => option.id === RT_VOICE)) {
				realtimeVoiceOptions = [{ id: RT_VOICE, name: RT_VOICE }, ...realtimeVoiceOptions];
			}
		}
	});
</script>

<div>
	<div class=" mt-0.5 mb-2.5 text-base font-medium">
		{$i18n.t('settings.admin.audio.sections.realtimeApi.title')}
	</div>

	<hr class=" border-gray-100/30 dark:border-gray-850/30 my-2" />

	<div class="mb-2 py-0.5 flex w-full justify-between">
		<div class=" self-center text-xs font-medium">
			{$i18n.t('settings.admin.audio.realtime.realtimeProvider.label')}
		</div>
		<div class="flex items-center relative">
			<select
				class="cursor-pointer w-fit pr-8 rounded-sm px-2 p-1 text-xs bg-transparent outline-hidden text-right"
				bind:value={RT_ENGINE}
			>
				<option value="">{$i18n.t('Disabled')}</option>
				<option value="openai">{$i18n.t('OpenAI Realtime')}</option>
			</select>
		</div>
	</div>

	{#if RT_ENGINE === 'openai'}
		{#if RT_TASK_MODEL_WARNING}
			<div
				class="mb-2 p-2 rounded-lg bg-yellow-50 dark:bg-yellow-900/30 border border-yellow-200 dark:border-yellow-800 text-xs text-yellow-800 dark:text-yellow-200"
			>
				{$i18n.t(
					'Realtime mode is enabled, but TASK_MODEL_EXTERNAL is not set. Background tasks (title, tags, follow-ups, context summarization) may fail for realtime-only models. Configure it in Admin > Settings > General.'
				)}
			</div>
		{/if}
		<div>
			<div class="mb-1.5 text-xs font-medium">
				{$i18n.t('settings.admin.audio.realtime.realtimeApiBaseUrl.label')}
			</div>
			<div class="mt-1 flex gap-2 mb-1">
				<input
					class="flex-1 w-full bg-transparent outline-hidden"
					placeholder={$i18n.t('API Base URL')}
					bind:value={RT_API_BASE_URL}
					required
				/>

				<SensitiveInput placeholder={$i18n.t('API Key')} bind:value={RT_API_KEY} />
			</div>
			<div class="mb-2 text-xs text-gray-400 dark:text-gray-500">
				{$i18n.t('settings.admin.audio.realtime.realtimeApiBaseUrl.description')}
			</div>
		</div>

		<div class="mb-2">
			<div class="mb-1 text-xs font-medium">
				{$i18n.t('settings.admin.audio.realtime.modelIds.label')}
			</div>

			{#if RT_MODELS.length > 0}
				<ul class="flex flex-col mb-1">
					{#each RT_MODELS as modelId, idx}
						<li class="flex gap-2 w-full justify-between items-center">
							<div class="text-sm flex-1 py-1 rounded-lg">{modelId}</div>
							<div class="shrink-0">
								<button
									aria-label={$i18n.t('Remove {{MODELID}} from list.', {
										MODELID: modelId
									})}
									type="button"
									on:click={() => {
										RT_MODELS = RT_MODELS.filter((_, i) => i !== idx);
									}}
								>
									<Minus strokeWidth="2" className="size-3.5" />
								</button>
							</div>
						</li>
					{/each}
				</ul>
			{:else}
				<div class="text-gray-500 text-xs text-center py-2">
					{$i18n.t('settings.admin.audio.realtime.modelIds.description')}
				</div>
			{/if}

			<div class="flex items-center gap-1">
				<input
					class="w-full py-1 text-sm rounded-lg bg-transparent placeholder:text-gray-300 dark:placeholder:text-gray-700 outline-hidden"
					bind:value={RT_MODEL_INPUT}
					placeholder={$i18n.t('Add a model ID (e.g. gpt-realtime-1.5)')}
					on:keydown={(e) => {
						if (e.key === 'Enter') {
							e.preventDefault();
							if (RT_MODEL_INPUT.trim() && !RT_MODELS.includes(RT_MODEL_INPUT.trim())) {
								RT_MODELS = [...RT_MODELS, RT_MODEL_INPUT.trim()];
								RT_MODEL_INPUT = '';
							}
						}
					}}
				/>
				<button
					type="button"
					aria-label={$i18n.t('Add')}
					on:click={() => {
						if (RT_MODEL_INPUT.trim() && !RT_MODELS.includes(RT_MODEL_INPUT.trim())) {
							RT_MODELS = [...RT_MODELS, RT_MODEL_INPUT.trim()];
							RT_MODEL_INPUT = '';
						}
					}}
				>
					<Plus className="size-3.5" strokeWidth="2" />
				</button>
			</div>
		</div>

		<div class="mb-2 py-0.5 flex w-full justify-between">
			<div class=" self-center text-xs font-medium">
				{$i18n.t('settings.admin.audio.realtime.voice.label')}
			</div>
			<div class="flex items-center relative">
				<select
					class="cursor-pointer w-fit pr-8 rounded-sm px-2 p-1 text-xs bg-transparent outline-hidden text-right"
					bind:value={RT_VOICE}
				>
					{#each realtimeVoiceOptions as voice}
						<option value={voice.id}>{voice.name}</option>
					{/each}
				</select>
			</div>
		</div>
		<div class="mb-2 text-xs text-gray-400 dark:text-gray-500">
			{$i18n.t('settings.admin.audio.realtime.voice.description')}
		</div>

		<div class="mb-2 py-0.5 flex w-full justify-between gap-3">
			<div class="self-center text-xs font-medium">
				{$i18n.t('settings.admin.audio.realtime.speechSpeed.label')}
			</div>
			<input
				type="number"
				step="0.05"
				min="0.25"
				max="1.5"
				class="w-24 rounded-lg py-1 px-2 text-right text-xs bg-gray-50 dark:text-gray-300 dark:bg-gray-850 outline-hidden"
				bind:value={RT_SPEED}
			/>
		</div>
		<div class="mb-2 text-xs text-gray-400 dark:text-gray-500">
			{$i18n.t('settings.admin.audio.realtime.speechSpeed.description')}
		</div>

		<div class="mb-2 py-0.5 flex w-full justify-between">
			<div class=" self-center text-xs font-medium">
				{$i18n.t('settings.admin.audio.realtime.voiceActivityDetectionVad.label')}
			</div>
			<div class="flex items-center relative">
				<select
					class="cursor-pointer w-fit pr-8 rounded-sm px-2 p-1 text-xs bg-transparent outline-hidden text-right"
					bind:value={RT_VAD_TYPE}
				>
					<option value="semantic_vad">{$i18n.t('Smart (AI-powered)')}</option>
					<option value="server_vad">{$i18n.t('Standard (volume-based)')}</option>
					<option value="push_to_talk">{$i18n.t('Push to Talk')}</option>
				</select>
			</div>
		</div>
		<div class="mb-2 text-xs text-gray-400 dark:text-gray-500">
			{$i18n.t('settings.admin.audio.realtime.voiceActivityDetectionVad.description')}
		</div>

		{#if RT_VAD_TYPE === 'semantic_vad'}
			<div class="mb-2 py-0.5 flex w-full justify-between">
				<div class=" self-center text-xs font-medium">
					{$i18n.t('settings.admin.audio.realtime.responseEagerness.label')}
				</div>
				<div class="flex items-center relative">
					<select
						class="cursor-pointer w-fit pr-8 rounded-sm px-2 p-1 text-xs bg-transparent outline-hidden text-right"
						bind:value={RT_SEMANTIC_VAD_EAGERNESS}
					>
						<option value="auto">{$i18n.t('Auto')}</option>
						<option value="low">{$i18n.t('Low')}</option>
						<option value="medium">{$i18n.t('Medium')}</option>
						<option value="high">{$i18n.t('High')}</option>
					</select>
				</div>
			</div>
			<div class="mb-2 text-xs text-gray-400 dark:text-gray-500">
				{$i18n.t('settings.admin.audio.realtime.responseEagerness.description')}
			</div>
		{:else if RT_VAD_TYPE === 'server_vad'}
			<div class="mb-2 py-0.5 flex w-full justify-between gap-3">
				<div class="self-center text-xs font-medium">
					{$i18n.t('settings.admin.audio.realtime.volumeThreshold.label')}
				</div>
				<input
					type="number"
					step="0.1"
					min="0"
					max="1"
					class="w-24 rounded-lg py-1 px-2 text-right text-xs bg-gray-50 dark:text-gray-300 dark:bg-gray-850 outline-hidden"
					bind:value={RT_SERVER_VAD_THRESHOLD}
				/>
			</div>
			<div class="mb-2 text-xs text-gray-400 dark:text-gray-500">
				{$i18n.t('settings.admin.audio.realtime.volumeThreshold.description')}
			</div>

			<div class="mb-2 py-0.5 flex w-full justify-between gap-3">
				<div class="self-center text-xs font-medium">
					{$i18n.t('settings.admin.audio.realtime.silenceDurationMs.label')}
				</div>
				<input
					type="number"
					min="0"
					class="w-24 rounded-lg py-1 px-2 text-right text-xs bg-gray-50 dark:text-gray-300 dark:bg-gray-850 outline-hidden"
					bind:value={RT_SERVER_VAD_SILENCE_DURATION_MS}
				/>
			</div>
			<div class="mb-2 text-xs text-gray-400 dark:text-gray-500">
				{$i18n.t('settings.admin.audio.realtime.silenceDurationMs.description')}
			</div>

			<div class="mb-2 py-0.5 flex w-full justify-between gap-3">
				<div class="self-center text-xs font-medium">
					{$i18n.t('settings.admin.audio.realtime.prefixPaddingMs.label')}
				</div>
				<input
					type="number"
					min="0"
					class="w-24 rounded-lg py-1 px-2 text-right text-xs bg-gray-50 dark:text-gray-300 dark:bg-gray-850 outline-hidden"
					bind:value={RT_SERVER_VAD_PREFIX_PADDING_MS}
				/>
			</div>
			<div class="mb-2 text-xs text-gray-400 dark:text-gray-500">
				{$i18n.t('settings.admin.audio.realtime.prefixPaddingMs.description')}
			</div>
		{/if}

		<hr class="border-gray-100/30 dark:border-gray-850/30 my-2" />

		<div class="flex items-center justify-between mb-2">
			<div class="text-xs font-medium">
				{$i18n.t('settings.admin.audio.realtime.autoRespondAfterSpeech.label')}
			</div>
			<Switch bind:state={RT_VAD_CREATE_RESPONSE} />
		</div>
		<div class="mb-2 text-xs text-gray-400 dark:text-gray-500">
			{$i18n.t('settings.admin.audio.realtime.autoRespondAfterSpeech.description')}
		</div>

		<div class="flex items-center justify-between mb-2">
			<div class="text-xs font-medium">
				{$i18n.t('settings.admin.audio.realtime.allowSpeechInterruption.label')}
			</div>
			<Switch bind:state={RT_VAD_INTERRUPT_RESPONSE} />
		</div>
		<div class="mb-2 text-xs text-gray-400 dark:text-gray-500">
			{$i18n.t('settings.admin.audio.realtime.allowSpeechInterruption.description')}
		</div>

		<div class="mb-2">
			<div class="mb-1.5 text-xs font-medium">
				{$i18n.t('settings.admin.audio.realtime.reEngageTimeoutMs.label')}
			</div>
			<div class="flex w-full">
				<div class="flex-1">
					<input
						class="w-full rounded-lg py-2 px-4 text-sm bg-gray-50 dark:text-gray-300 dark:bg-gray-850 outline-hidden"
						bind:value={RT_VAD_IDLE_TIMEOUT_MS}
						placeholder={$i18n.t('Leave empty to disable')}
					/>
				</div>
			</div>
			<div class="mt-1 text-xs text-gray-400 dark:text-gray-500">
				{$i18n.t('settings.admin.audio.realtime.reEngageTimeoutMs.description')}
			</div>
		</div>

		<hr class="border-gray-100/30 dark:border-gray-850/30 my-2" />

		<div class="mb-2 py-0.5 flex w-full justify-between">
			<div class=" self-center text-xs font-medium">
				{$i18n.t('settings.admin.audio.realtime.transcriptionModel.label')}
			</div>
			<div class="flex items-center relative">
				<select
					class="cursor-pointer w-fit pr-8 rounded-sm px-2 p-1 text-xs bg-transparent outline-hidden text-right"
					bind:value={RT_TRANSCRIPTION_MODEL}
				>
					<option value="gpt-4o-transcribe">{$i18n.t('gpt-4o-transcribe')}</option>
					<option value="gpt-4o-mini-transcribe">{$i18n.t('gpt-4o-mini-transcribe')}</option>
					<option value="whisper-1">{$i18n.t('whisper-1')}</option>
				</select>
			</div>
		</div>
		<div class="mb-2 text-xs text-gray-400 dark:text-gray-500">
			{$i18n.t('settings.admin.audio.realtime.transcriptionModel.description')}
		</div>

		<div class="mb-2">
			<div class="mb-1.5 text-xs font-medium">
				{$i18n.t('settings.admin.audio.realtime.transcriptionPrompt.label')}
			</div>
			<div class="flex w-full">
				<div class="flex-1">
					<input
						class="w-full rounded-lg py-2 px-4 text-sm bg-gray-50 dark:text-gray-300 dark:bg-gray-850 outline-hidden"
						bind:value={RT_TRANSCRIPTION_PROMPT}
						placeholder={$i18n.t('Optional guidance prompt for transcription')}
					/>
				</div>
			</div>
			<div class="mt-1 text-xs text-gray-400 dark:text-gray-500">
				{$i18n.t('settings.admin.audio.realtime.transcriptionPrompt.description')}
				<a
					class="ml-1 underline"
					href="https://platform.openai.com/docs/guides/speech-to-text#prompting"
					target="_blank"
					rel="noreferrer"
				>
					{$i18n.t('OpenAI docs')}
				</a>
			</div>
		</div>

		<div class="mb-2 py-0.5 flex w-full justify-between">
			<div class=" self-center text-xs font-medium">
				{$i18n.t('settings.admin.audio.realtime.noiseReduction.label')}
			</div>
			<div class="flex items-center relative">
				<select
					class="cursor-pointer w-fit pr-8 rounded-sm px-2 p-1 text-xs bg-transparent outline-hidden text-right"
					bind:value={RT_NOISE_REDUCTION}
				>
					<option value="near_field">{$i18n.t('Close range (headset/laptop)')}</option>
					<option value="far_field">{$i18n.t('Far range (room mic/speaker)')}</option>
					<option value="">{$i18n.t('None')}</option>
				</select>
			</div>
		</div>
		<div class="mb-2 text-xs text-gray-400 dark:text-gray-500">
			{$i18n.t('settings.admin.audio.realtime.noiseReduction.description')}
		</div>

		<hr class="border-gray-100/30 dark:border-gray-850/30 my-2" />

		<div class="mb-2">
			<div class=" mb-1.5 text-xs font-medium">
				{$i18n.t('settings.admin.audio.realtime.maxOutputTokens.label')}
			</div>
			<div class="flex w-full">
				<div class="flex-1">
					<input
						type="number"
						min="-1"
						max="4096"
						class="w-full rounded-lg py-2 px-4 text-sm bg-gray-50 dark:text-gray-300 dark:bg-gray-850 outline-hidden"
						bind:value={RT_MAX_RESPONSE_OUTPUT_TOKENS}
						placeholder={$i18n.t('Leave empty for the model default')}
					/>
				</div>
			</div>
			<div class="mt-1 text-xs text-gray-400 dark:text-gray-500">
				{$i18n.t('settings.admin.audio.realtime.maxOutputTokens.description')}
			</div>
		</div>

		<hr class="border-gray-100/30 dark:border-gray-850/30 my-2" />

		<div class="grid grid-cols-1 sm:grid-cols-2 gap-2 mb-2">
			<div>
				<div class="mb-2 py-0.5 flex w-full justify-between gap-3">
					<div class="self-center text-xs font-medium">
						{$i18n.t('settings.admin.audio.realtime.sessionIdleTimeoutSeconds.label')}
					</div>
					<input
						type="number"
						min="60"
						max="1800"
						class="w-24 rounded-lg py-1 px-2 text-right text-xs bg-gray-50 dark:text-gray-300 dark:bg-gray-850 outline-hidden"
						bind:value={RT_SESSION_TIMEOUT}
					/>
				</div>
				<div class="text-xs text-gray-400 dark:text-gray-500">
					{$i18n.t('settings.admin.audio.realtime.sessionIdleTimeoutSeconds.description')}
				</div>
			</div>
			<div>
				<div class="mb-2 py-0.5 flex w-full justify-between gap-3">
					<div class="self-center text-xs font-medium">
						{$i18n.t('settings.admin.audio.realtime.idleCallCheckInIntervalSeconds.label')}
					</div>
					<input
						type="number"
						min="0"
						max="300"
						class="w-24 rounded-lg py-1 px-2 text-right text-xs bg-gray-50 dark:text-gray-300 dark:bg-gray-850 outline-hidden"
						bind:value={RT_IDLE_CALL_CHECKIN_INTERVAL}
					/>
				</div>
				<div class="text-xs text-gray-400 dark:text-gray-500">
					{$i18n.t('settings.admin.audio.realtime.idleCallCheckInIntervalSeconds.description')}
				</div>
			</div>
		</div>

		<div class="mb-2">
			<div class="mb-1.5 text-xs font-medium">
				{$i18n.t('settings.admin.audio.realtime.idleCallCheckInPrompt.label')}
			</div>
			<Textarea
				className="w-full rounded-lg py-2 px-4 text-sm bg-gray-50 dark:text-gray-300 dark:bg-gray-850 outline-hidden resize-y"
				bind:value={RT_IDLE_CALL_CHECKIN_PROMPT}
				rows={4}
			/>
			<div class="mt-1 text-xs text-gray-400 dark:text-gray-500">
				{$i18n.t('settings.admin.audio.realtime.idleCallCheckInPrompt.description')}
			</div>
		</div>

		<div class="mb-2">
			<div class="mb-2 py-0.5 flex w-full justify-between">
				<div class="self-center text-xs font-medium">
					{$i18n.t('settings.admin.audio.realtime.truncationStrategy.label')}
				</div>
				<div class="flex items-center relative">
					<select
						class="cursor-pointer w-fit pr-8 rounded-sm px-2 p-1 text-xs bg-transparent outline-hidden text-right"
						bind:value={RT_TRUNCATION_STRATEGY}
					>
						<option value="auto">{$i18n.t('Auto (OpenAI default)')}</option>
						<option value="retention_ratio">{$i18n.t('Retain After Truncation')}</option>
						<option value="disabled">{$i18n.t('Disabled')}</option>
					</select>
				</div>
			</div>
			<div class="mt-1 text-xs text-gray-400 dark:text-gray-500">
				{$i18n.t('settings.admin.audio.realtime.truncationStrategy.description')}
			</div>
		</div>

		{#if RT_TRUNCATION_STRATEGY === 'retention_ratio'}
			<div class="mb-2 py-0.5 flex w-full justify-between gap-3">
				<div class="self-center text-xs font-medium">
					{$i18n.t('settings.admin.audio.realtime.retainAfterTruncation.label')}
				</div>
				<input
					type="number"
					step="0.05"
					min="0"
					max="1"
					class="w-24 rounded-lg py-1 px-2 text-right text-xs bg-gray-50 dark:text-gray-300 dark:bg-gray-850 outline-hidden"
					bind:value={RT_TRUNCATION_RETENTION_RATIO}
				/>
			</div>
			<div class="mb-2 text-xs text-gray-400 dark:text-gray-500">
				{$i18n.t('settings.admin.audio.realtime.retainAfterTruncation.description')}
			</div>

			<div class="mb-2 py-0.5 flex w-full justify-between gap-3">
				<div class="self-center text-xs font-medium">
					{$i18n.t('settings.admin.audio.realtime.conversationTokenLimit.label')}
				</div>
				<input
					type="number"
					min="0"
					class="w-28 rounded-lg py-1 px-2 text-right text-xs bg-gray-50 dark:text-gray-300 dark:bg-gray-850 outline-hidden"
					bind:value={RT_TRUNCATION_TOKEN_LIMIT}
					placeholder={$i18n.t('No limit')}
				/>
			</div>
			<div class="mb-2 text-xs text-gray-400 dark:text-gray-500">
				{$i18n.t('settings.admin.audio.realtime.conversationTokenLimit.description')}
			</div>
		{/if}

		<hr class="border-gray-100/30 dark:border-gray-850/30 my-2" />

		<div class="flex items-center justify-between mb-2">
			<div class="text-sm font-medium">
				{$i18n.t('settings.admin.audio.realtime.conversationContext.label')}
			</div>
			<Switch bind:state={RT_CONTEXT_ENABLED} />
		</div>
		<div class="mb-2 text-xs text-gray-400 dark:text-gray-500">
			{$i18n.t('settings.admin.audio.realtime.conversationContext.description')}
		</div>

		{#if RT_CONTEXT_ENABLED}
			<div class="text-xs text-gray-400 dark:text-gray-500 mb-2">
				{$i18n.t(
					'Adds conversation continuity on top of the system instructions. Recent exchanges can be replayed directly, and older history can optionally be summarized separately.'
				)}
			</div>

			<div class="grid grid-cols-1 sm:grid-cols-2 gap-2 mb-2">
				<div>
					<div class="mb-2 py-0.5 flex w-full justify-between gap-3">
						<div class="self-center text-xs font-medium">
							{$i18n.t('settings.admin.audio.realtime.recentExchangesToReplay.label')}
						</div>
						<input
							type="number"
							class="w-24 rounded-lg py-1 px-2 text-right text-xs bg-gray-50 dark:text-gray-300 dark:bg-gray-850 outline-hidden"
							bind:value={RT_CONTEXT_RECENT_EXCHANGES_LIMIT}
							min="1"
							max="100"
						/>
					</div>
					<div class="text-xs text-gray-400 dark:text-gray-500">
						{$i18n.t('settings.admin.audio.realtime.recentExchangesToReplay.description')}
					</div>
				</div>
				<div>
					<div class=" mb-1.5 text-xs font-medium">
						{$i18n.t('settings.admin.audio.realtime.unansweredLastUserTurn.label')}
					</div>
					<select
						class="w-full rounded-lg py-2 px-4 text-sm bg-gray-50 dark:text-gray-300 dark:bg-gray-850 outline-hidden"
						bind:value={RT_CONTEXT_UNANSWERED_LAST_USER_TURN}
					>
						<option value="discard">{$i18n.t('Discard unanswered last user turn')}</option>
						<option value="replay">{$i18n.t('Replay unanswered last user turn')}</option>
					</select>
					<div class="mt-1 text-xs text-gray-400 dark:text-gray-500">
						{$i18n.t('settings.admin.audio.realtime.unansweredLastUserTurn.description')}
					</div>
				</div>
			</div>

			<div class="flex items-center justify-between mb-2">
				<div class="text-xs font-medium">
					{$i18n.t('settings.admin.audio.realtime.summarizeOlderHistory.label')}
				</div>
				<Switch bind:state={RT_CONTEXT_SUMMARIZE} />
			</div>

			{#if RT_CONTEXT_SUMMARIZE}
				<div class="text-xs text-gray-400 dark:text-gray-500 mb-2">
					{$i18n.t('settings.admin.audio.realtime.summarizeOlderHistory.description')}
				</div>

				<div class="grid grid-cols-1 sm:grid-cols-2 gap-2 mb-2">
					<div>
						<div class="mb-2 py-0.5 flex w-full justify-between gap-3">
							<div class="self-center text-xs font-medium">
								{$i18n.t('settings.admin.audio.realtime.olderExchangesToSummarize.label')}
							</div>
							<input
								type="number"
								class="w-24 rounded-lg py-1 px-2 text-right text-xs bg-gray-50 dark:text-gray-300 dark:bg-gray-850 outline-hidden"
								bind:value={RT_CONTEXT_MAX_HISTORY_EXCHANGES}
								min="1"
								max="200"
							/>
						</div>
						<div class="text-xs text-gray-400 dark:text-gray-500">
							{$i18n.t('settings.admin.audio.realtime.olderExchangesToSummarize.description')}
						</div>
					</div>
					<div>
						<div class="mb-2 py-0.5 flex w-full justify-between gap-3">
							<div class="self-center text-xs font-medium">
								{$i18n.t('settings.admin.audio.realtime.olderHistoryTextBudgetBytes.label')}
							</div>
							<input
								type="number"
								class="w-28 rounded-lg py-1 px-2 text-right text-xs bg-gray-50 dark:text-gray-300 dark:bg-gray-850 outline-hidden"
								bind:value={RT_CONTEXT_MAX_HISTORY_BYTES}
								min="100"
								max="100000"
							/>
						</div>
						<div class="text-xs text-gray-400 dark:text-gray-500">
							{$i18n.t('settings.admin.audio.realtime.olderHistoryTextBudgetBytes.description')}
						</div>
					</div>
				</div>

				<div class="mb-2 py-0.5 flex w-full justify-between gap-3">
					<div class="self-center text-xs font-medium">
						{$i18n.t('settings.admin.audio.realtime.summaryTargetSizeCharacters.label')}
					</div>
					<input
						type="number"
						class="w-28 rounded-lg py-1 px-2 text-right text-xs bg-gray-50 dark:text-gray-300 dark:bg-gray-850 outline-hidden"
						bind:value={RT_CONTEXT_SUMMARY_MAX_SIZE}
						min="100"
						max="10000"
					/>
				</div>
				<div class="mb-2 text-xs text-gray-400 dark:text-gray-500">
					{$i18n.t('settings.admin.audio.realtime.summaryTargetSizeCharacters.description')}
				</div>

				<div>
					<div class=" mb-1.5 text-xs font-medium">
						{$i18n.t('settings.admin.audio.realtime.contextSummaryPrompt.label')}
					</div>
					<Tooltip
						content={$i18n.t('Leave empty to use the default prompt, or enter a custom prompt')}
						placement="top-start"
						className="w-full"
					>
						<Textarea
							className="w-full rounded-lg py-2 px-4 text-sm bg-gray-50 dark:text-gray-300 dark:bg-gray-850 outline-hidden resize-y"
							bind:value={RT_CONTEXT_SUMMARY_PROMPT}
							placeholder={$i18n.t(
								'Leave empty to use the default prompt, or enter a custom prompt'
							)}
							rows={6}
						/>
					</Tooltip>
					<div class="mt-1 text-xs text-gray-400 dark:text-gray-500">
						{$i18n.t('settings.admin.audio.realtime.contextSummaryPrompt.description')}
					</div>
				</div>
			{/if}
		{/if}
	{/if}
</div>
