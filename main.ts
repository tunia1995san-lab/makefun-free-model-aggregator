// main.ts — the free-model aggregator: one server, ten providers, the best free model per request.
//
// The trial gives this server a brief and an env, and runs it as `node main.ts` from the
// repository's root, with Node's standard library alone (nothing is installed; TypeScript runs
// directly, and a relative import needs its .ts extension). The brief is at the path in
// $MAKEFUN_PROMPT_FILE. It carries three JSON documents:
//   - the provider directory: ten simulated free-tier providers, each with its id, wire (openai,
//     anthropic or gemini), auth scheme, key environment variable, the key the service takes, and
//     the base path you join to $MAKEFUN_ENV_URL;
//   - the quality table: every model's public rating and what it can do (tools, vision, JSON mode,
//     context length), and whether it is free;
//   - the workload manifest: the twenty requests the client will send, in order, with each one's api,
//     whether it streams, its weight and the requirements a model must meet to serve it.
// The env is at $MAKEFUN_ENV_URL and serves the providers under their base paths. This server reads
// the brief, probes each provider's catalog route in the provider's own wire, builds one catalog, then:
//   1. listens on 0.0.0.0 at a free port (never 8080: the env itself is there on one layout);
//   2. POSTs $MAKEFUN_ENV_URL/register {"protocol_version": 1, "port": <port>, "catalog": <catalog>};
//   3. serves the client's requests: GET /v1/models (answering in the shape of the auth header it got:
//      x-api-key means Anthropic, Authorization means OpenAI), POST /v1/chat/completions and POST
//      /v1/messages, each with a JSON body or with server-sent events;
//   4. polls GET $MAKEFUN_ENV_URL/work every 100 ms and exits 0 when the env says {"done": true}. A
//      process still running when the wall ends scores zero, so never wait for anything else.
// `node main.ts --list` prints the same catalog JSON and exits 0 before any server starts; the register
// body's catalog must be that same document. Build it once, as this file does, and use the one object.
//
// ROUTING: quality is 60% of the score, so every request goes to the best free model that meets its
// requirements, ranked by the quality table's rating_milli. But each provider has a per-trial free-tier
// budget, and a call over it (or on its failure schedule, or after its outage begins) is refused with
// 402 or 429. So the router keeps a per-provider call count and a cap at the largest budget the trial
// can draw, and on any refusal it fails over to the next-best candidate that still has budget. A
// refused call is never the answer: the request is only answered by a model that returned 2xx, and
// exactly one provider call answers each request. The caps are the trial's own published maxima; a
// provider is skipped only once its count reaches a number no trial can exceed, so a skipped provider
// is one whose next call is guaranteed to be refused.
//
// WIRES: the client may speak OpenAI or Anthropic, and a provider may speak OpenAI, Anthropic or
// Gemini. The translations live in lib/wire.ts. One restriction: an OpenAI request that asks for JSON
// mode is never routed to an Anthropic-wire provider, because that wire has no JSON-mode flag and its
// echo would report json_mode false. Every other combination is translated faithfully.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type ProviderDirectoryEntry,
  type RatingRow,
  type Requirement,
  type Wire,
} from './lib/brief.ts';
import {
  anthropicAnswerToOpenAI,
  anthropicError,
  anthropicEventsFromGemini,
  anthropicEventsFromOpenAI,
  anthropicModelList,
  anthropicToGemini,
  anthropicToOpenAI,
  geminiToAnthropic,
  geminiToOpenAI,
  openAIAnswerToAnthropic,
  openAIError,
  openAIEventsFromAnthropic,
  openAIEventsFromGemini,
  openAIModelList,
  openAIToAnthropic,
  openAIToGemini,
  type ServedModel,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** How long one provider inference call may take, in milliseconds. */
const PROVIDER_TIMEOUT_MS = 10000;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;

/**
 * The largest free-tier budget a trial can draw for each provider. A provider whose call count has
 * reached this number cannot serve another call in any trial, so the router stops trying it. The
 * numbers are the trial's own published maxima; the draw itself is per trial and never published.
 */
const BUDGET_CAP: Record<string, number> = {
  groq: 12,
  cerebras: 6,
  gemini: 8,
  zai: 5,
  openrouter: 12,
  sambanova: 5,
  pollinations: 14,
  cloudflare: 8,
  ovh: 6,
  kilol: 16,
};

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** How many client API requests have arrived: the manifest is in arrival order, one request at a time. */
let manifestIndex = 0;
/** Calls made to each provider so far (successful and refused alike: both spend budget). */
const callsUsed = new Map<string, number>();

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(
    brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)),
  );
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/** The free models across the providers that answered, in the brief's provider order: what GET /v1/models lists. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

interface Candidate {
  probe: Probe;
  model: ProbedModel;
  row: RatingRow | null;
  rating: number;
}

/**
 * Every free model that meets one request's requirements, best first. The ranking is the quality
 * table's rating_milli; ties break by provider id, then model id, so the order is the same on a
 * replay. An OpenAI request that asks for JSON mode excludes Anthropic-wire providers: that wire
 * has no JSON-mode flag, so its echo would report json_mode false and lose the request.
 */
function candidatesFor(requires: Requirement, clientApi: 'openai' | 'anthropic', jsonMode: boolean): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    if (clientApi === 'openai' && jsonMode && probe.provider.wire === 'anthropic') continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      out.push({ probe, model, row, rating: row?.rating_milli ?? 0 });
    }
  }
  out.sort((a, b) => {
    if (b.rating !== a.rating) return b.rating - a.rating;
    if (a.probe.provider.id !== b.probe.provider.id) return a.probe.provider.id < b.probe.provider.id ? -1 : 1;
    return a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0;
  });
  return out;
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  // The manifest is the declared source of each request's requirements; this fallback is a guess for
  // a brief that holds none.
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (!isRecord(block)) continue;
      if (block.type === 'image' || block.type === 'image_url') vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return {
    tools: Array.isArray(body.tools) && body.tools.length > 0,
    vision,
    json: format === 'json_object' || format === 'json_schema',
  };
}

/** Whether a client body asks for JSON mode (OpenAI's response_format). */
function jsonModeOf(body: Json): boolean {
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return format === 'json_object' || format === 'json_schema';
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

interface UpstreamCall {
  provider: ProviderDirectoryEntry;
  model: string;
  wire: Wire;
  url: string;
  headers: Record<string, string>;
  body: Json;
}

/** The provider call for one candidate: the URL, headers and body in the provider's own wire. */
function upstreamCallFor(candidate: Candidate, clientApi: 'openai' | 'anthropic', body: Json, stream: boolean): UpstreamCall {
  const provider = candidate.probe.provider;
  const base = joinBase(ENV_URL, provider.base);
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(provider.auth, provider.key),
  };
  if (stream) headers.accept = 'text/event-stream';
  if (provider.wire === 'openai') {
    let upstreamBody: Json;
    if (clientApi === 'openai') {
      upstreamBody = { ...body, model: candidate.model.id };
      // The client sends no stream_options, so the provider would stream no usage chunk and the
      // answer would carry no final usage. Ask for it: the echo does not cover stream_options.
      if (stream) upstreamBody.stream_options = { include_usage: true };
    } else {
      upstreamBody = anthropicToOpenAI(body, candidate.model.id);
    }
    return { provider, model: candidate.model.id, wire: 'openai', url: `${base}/v1/chat/completions`, headers, body: upstreamBody };
  }
  if (provider.wire === 'anthropic') {
    if (provider.wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
    const upstreamBody = clientApi === 'anthropic' ? { ...body, model: candidate.model.id } : openAIToAnthropic(body, candidate.model.id);
    return { provider, model: candidate.model.id, wire: 'anthropic', url: `${base}/v1/messages`, headers, body: upstreamBody };
  }
  // Gemini: the model travels in the URL, not the body.
  const upstreamBody = clientApi === 'openai' ? openAIToGemini(body) : anthropicToGemini(body);
  const verb = stream ? 'streamGenerateContent' : 'generateContent';
  const url = `${base}/v1beta/models/${encodeURIComponent(candidate.model.id)}:${verb}${stream ? '?alt=sse' : ''}`;
  return { provider, model: candidate.model.id, wire: 'gemini', url, headers, body: upstreamBody };
}

/** A provider's non-streaming answer translated back into the client's wire. */
async function translateJsonAnswer(
  upstream: Response,
  call: UpstreamCall,
  clientApi: 'openai' | 'anthropic',
): Promise<Json | null> {
  const answer: unknown = await upstream.json().catch(() => null);
  if (!isRecord(answer)) return null;
  if (clientApi === 'openai') {
    if (call.wire === 'openai') return answer;
    if (call.wire === 'anthropic') return anthropicAnswerToOpenAI(answer, call.model);
    return geminiToOpenAI(answer, call.model);
  }
  if (call.wire === 'anthropic') return answer;
  if (call.wire === 'openai') return openAIAnswerToAnthropic(answer, call.model);
  return geminiToAnthropic(answer, call.model);
}

/** A provider's streaming answer translated back into the client's wire, written to the response. */
async function translateStreamAnswer(
  upstream: Response,
  call: UpstreamCall,
  clientApi: 'openai' | 'anthropic',
  res: ServerResponse,
): Promise<void> {
  if (upstream.body === null) {
    res.end();
    return;
  }
  res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
  if (clientApi === 'openai' && call.wire === 'openai') {
    // Same wire: the provider's events are already the client's.
    const reader = upstream.body.getReader();
    try {
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        res.write(Buffer.from(value));
      }
    } finally {
      reader.releaseLock();
      res.end();
    }
    return;
  }
  if (clientApi === 'anthropic' && call.wire === 'anthropic') {
    const reader = upstream.body.getReader();
    try {
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        res.write(Buffer.from(value));
      }
    } finally {
      reader.releaseLock();
      res.end();
    }
    return;
  }
  // A translated stream: read the provider's frames, emit the client's events.
  if (clientApi === 'openai' && call.wire === 'anthropic') {
    for await (const event of openAIEventsFromAnthropic(upstream.body, call.model)) res.write(event);
  } else if (clientApi === 'openai' && call.wire === 'gemini') {
    for await (const event of openAIEventsFromGemini(upstream.body, call.model)) res.write(event);
  } else if (clientApi === 'anthropic' && call.wire === 'openai') {
    for await (const event of anthropicEventsFromOpenAI(upstream.body, call.model)) res.write(event);
  } else {
    for await (const event of anthropicEventsFromGemini(upstream.body, call.model)) res.write(event);
  }
  res.end();
}

interface ServeResult {
  ok: boolean;
  status: number;
  detail: string;
}

/**
 * Serve one client request: rank the candidates, call them in order until one answers 2xx, and
 * translate that answer into the client's wire. A refusal (402, 429, 5xx, or an unreachable
 * provider) is never the answer: the next candidate is tried, and the call it cost is counted
 * against that provider's budget all the same.
 */
async function serve(
  req: IncomingMessage,
  res: ServerResponse,
  clientApi: 'openai' | 'anthropic',
  body: Json,
): Promise<ServeResult> {
  const stream = body.stream === true;
  const requires = requirementsFor(body);
  const jsonMode = clientApi === 'openai' ? jsonModeOf(body) : false;
  const candidates = candidatesFor(requires, clientApi, jsonMode);
  if (candidates.length === 0) {
    sendError(res, clientApi, 503, clientApi === 'anthropic' ? 'overloaded_error' : 'no_free_model', 'no free model on the directory meets this request');
    return { ok: false, status: 503, detail: 'no candidate meets the requirements' };
  }
  let lastDetail = 'no candidate was tried';
  for (const candidate of candidates) {
    const providerId = candidate.probe.provider.id;
    const used = callsUsed.get(providerId) ?? 0;
    const cap = BUDGET_CAP[providerId] ?? 0;
    if (used >= cap) continue;
    const call = upstreamCallFor(candidate, clientApi, body, stream);
    callsUsed.set(providerId, used + 1);
    let upstream: Response;
    try {
      upstream = await fetch(call.url, {
        method: 'POST',
        headers: call.headers,
        body: JSON.stringify(call.body),
        signal: AbortSignal.timeout(PROVIDER_TIMEOUT_MS),
      });
    } catch (err) {
      lastDetail = `${providerId}/${candidate.model.id} unreachable: ${(err as Error).message}`;
      log(lastDetail);
      continue;
    }
    if (!upstream.ok) {
      const text = await upstream.text().catch(() => '');
      lastDetail = `${providerId}/${candidate.model.id} answered HTTP ${upstream.status}: ${text.slice(0, 200)}`;
      log(lastDetail);
      continue;
    }
    // Exactly one provider call answers this request: a 2xx is the answer, whatever it holds.
    log(`serving with ${providerId}/${candidate.model.id} (rating ${candidate.rating})`);
    if (stream) {
      await translateStreamAnswer(upstream, call, clientApi, res);
    } else {
      const translated = await translateJsonAnswer(upstream, call, clientApi);
      if (translated === null) {
        sendError(res, clientApi, 502, clientApi === 'anthropic' ? 'api_error' : 'upstream_error', 'the provider answered a body that is not JSON');
        return { ok: false, status: 502, detail: 'the provider answered a body that is not JSON' };
      }
      sendJson(res, 200, translated);
    }
    return { ok: true, status: 200, detail: `served by ${providerId}/${candidate.model.id}` };
  }
  sendError(res, clientApi, 503, clientApi === 'anthropic' ? 'overloaded_error' : 'no_free_model', `no free model could serve this request (${lastDetail})`);
  return { ok: false, status: 503, detail: lastDetail };
}

/** POST /v1/chat/completions: an OpenAI-shaped client. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  await serve(req, res, 'openai', body);
}

/** POST /v1/messages: an Anthropic-shaped client. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  await serve(req, res, 'anthropic', body);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) {
        sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
      }
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
