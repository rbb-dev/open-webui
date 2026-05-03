import { AUDIO_API_BASE_URL } from '$lib/constants';

export type RealtimeClientConfig = {
	enabled: boolean;
	defaults: {
		voice?: string;
		vad_type?: 'semantic_vad' | 'server_vad' | 'push_to_talk';
		server_vad_threshold?: number;
		server_vad_silence_duration_ms?: number;
		server_vad_prefix_padding_ms?: number;
		semantic_vad_eagerness?: 'low' | 'medium' | 'high' | 'auto';
		noise_reduction?: 'near_field' | 'far_field' | '';
		speed?: number;
		vad_create_response?: boolean;
		vad_interrupt_response?: boolean;
		session_timeout?: number;
		idle_call_checkin_interval?: number;
	};
	capabilities: {
		models: string[];
		voices: { id: string; name: string }[];
		voices_by_model: Record<string, string[]>;
		vad_types: string[];
		noise_reduction: string[];
		semantic_vad_eagerness: string[];
	};
};

/**
 * Read an HTTP response body as JSON when possible, falling back to plain text.
 *
 * Standard frontend pattern (`if (!res.ok) throw await res.json()`) crashes with
 * a SyntaxError when the body is plain text — e.g., when Starlette/Uvicorn
 * returns "Internal Server Error" because an unhandled exception bypassed
 * FastAPI's exception machinery.  This helper inspects Content-Type first and
 * gracefully degrades, so the realtime overlay can show an actionable message.
 */
async function readResponseBody(
	res: Response
): Promise<{ data: any; isJson: boolean; raw: string }> {
	const ct = (res.headers.get('content-type') || '').toLowerCase();
	if (ct.includes('application/json')) {
		try {
			return { data: await res.json(), isJson: true, raw: '' };
		} catch (_) {
			/* fall through to text */
		}
	}
	const raw = await res.text();
	try {
		return { data: JSON.parse(raw), isJson: true, raw };
	} catch (_) {
		return { data: null, isJson: false, raw };
	}
}

export const negotiateRealtimeSdp = async (
	token: string,
	model: string,
	sdpOffer: string,
	options?: {
		tool_ids?: string[];
		tool_servers?: any[];
		features?: Record<string, any>;
		terminal_id?: string;
		chat_id?: string;
		session_id?: string;
		system_prompt?: string;
		language?: string;
	}
): Promise<{ sdp_answer: string; call_id: string }> => {
	let error: any = null;

	const res = await fetch(`${AUDIO_API_BASE_URL}/realtime/negotiate`, {
		method: 'POST',
		headers: {
			'Content-Type': 'application/json',
			Authorization: `Bearer ${token}`
		},
		body: JSON.stringify({
			model,
			sdp_offer: sdpOffer,
			...(options || {})
		})
	})
		.then(async (response) => {
			const { data, isJson, raw } = await readResponseBody(response);
			if (!response.ok) {
				if (isJson) throw data;
				throw {
					detail: raw?.trim()
						? `Server error (${response.status}): ${raw.slice(0, 200)}`
						: `Server error (${response.status})`,
					status: response.status
				};
			}
			if (!isJson) {
				throw {
					detail: 'Negotiate returned non-JSON response',
					status: response.status
				};
			}
			return data;
		})
		.catch((err) => {
			console.error(err);
			error = err;
			return null;
		});

	if (error) {
		throw error;
	}

	return res;
};

export const getRealtimeClientConfig = async (
	token: string = ''
): Promise<RealtimeClientConfig> => {
	let error: any = null;

	const res = await fetch(`${AUDIO_API_BASE_URL}/realtime/config`, {
		method: 'GET',
		headers: {
			'Content-Type': 'application/json',
			Authorization: `Bearer ${token}`
		}
	})
		.then(async (response) => {
			const { data, isJson, raw } = await readResponseBody(response);
			if (!response.ok) {
				if (isJson) throw data;
				throw {
					detail: raw?.trim()
						? `Server error (${response.status}): ${raw.slice(0, 200)}`
						: `Server error (${response.status})`,
					status: response.status
				};
			}
			if (!isJson) {
				throw {
					detail: 'Realtime config returned non-JSON response',
					status: response.status
				};
			}
			return data;
		})
		.catch((err) => {
			console.error(err);
			error = err?.detail ?? err;
			return null;
		});

	if (error) {
		throw error;
	}

	return res as RealtimeClientConfig;
};
