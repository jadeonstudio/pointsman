import { API_URL, MAX_RESPONSE_BYTES, VERSION, ControlError, fail } from './constants.mjs';

/** One HTTPS request. No retries, redirects, configurable origin, or payload logging. */
export async function callTypeSafe(payload, key, { timeoutMs, signal, fetchImpl = globalThis.fetch } = {}) {
  const controller = new AbortController();
  let timer;
  let rejectAbort;
  const stopped = new Promise((_, reject) => { rejectAbort = reject; });
  const cancel = () => { controller.abort(); rejectAbort(new ControlError('CANCELLED')); };
  if (signal?.aborted) fail('CANCELLED');
  signal?.addEventListener('abort', cancel, { once: true });
  timer = setTimeout(() => {
    controller.abort(); rejectAbort(new ControlError('TIMEOUT'));
  }, timeoutMs);
  const send = async () => {
    const response = await fetchImpl(API_URL, {
      method: 'POST', headers: { Authorization: `Bearer ${key}`, 'Content-Type': 'application/json', Accept: 'application/json', 'User-Agent': `pointsman/${VERSION}` },
      body: JSON.stringify(payload), signal: controller.signal, redirect: 'error',
    });
    if (!response.ok) {
      await response.body?.cancel().catch(() => {});
      if (response.status === 401 || response.status === 403) fail('AUTH_ERROR');
      if (response.status === 429) fail('RATE_LIMITED');
      if (response.status >= 500) fail('PROVIDER_UNAVAILABLE');
      fail('PROVIDER_REJECTED');
    }
    if (!(response.headers.get('content-type') || '').toLowerCase().includes('application/json')) fail('MALFORMED_RESPONSE');
    const reader = response.body?.getReader();
    if (!reader) fail('MALFORMED_RESPONSE');
    const chunks = [];
    let bytes = 0;
    try {
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        bytes += value.byteLength;
        if (bytes > MAX_RESPONSE_BYTES) fail('RESPONSE_TOO_LARGE');
        chunks.push(Buffer.from(value));
      }
      try { return JSON.parse(Buffer.concat(chunks).toString('utf8')); }
      catch { fail('MALFORMED_RESPONSE'); }
    } finally { await reader.cancel().catch(() => {}); reader.releaseLock(); }
  };
  try { return await Promise.race([send(), stopped]); }
  catch (e) { if (e instanceof ControlError) throw e; fail('NETWORK_ERROR'); }
  finally { clearTimeout(timer); signal?.removeEventListener('abort', cancel); }
}
