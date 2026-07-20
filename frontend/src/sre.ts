// SRE broker client: talks only to OUR backend proxy (/api/sre/*), which
// replays the terminal's own requests. Credentials live in sessionStorage
// (gone when the tab closes) and travel in POST bodies, never URLs.
import { rangeToQuery, type RangeQuery } from './api'
import { authFetch } from './auth'
import type { IfBreakout } from './types'

export interface SreHolding {
  broker?: 'SRE' | 'Groww'
  symbol: string | null
  series: string | null
  yahoo_symbol: string | null
  name: string | null
  exchange: string | null
  isin: string | null
  token_id: string | null
  quantity: number | null
  avg_price: number | null
  ltp: number | null
  prev_close: number | null
  pledged_qty: number | null
  invested: number | null
  current_value: number | null
  pnl: number | null
  pnl_pct: number | null
  day_pnl: number | null
  day_change_pct: number | null
  raw: Record<string, unknown>
}

export interface SreTokenInfo {
  client: string | null
  expires_at: string | null
  seconds_left: number | null
  expired: boolean
}

export interface HoldingsResponse {
  holdings: SreHolding[]
  count: number
  token: SreTokenInfo | null
  raw: unknown
}

export interface SignalVerdict {
  verdict: string
  reason: string
}

export interface SignalLevel {
  price: number | null
  distance_pct: number | null
  touches: number | null
  strength: number | null
}

export interface CompactSignal {
  symbol: string
  last_date: string | null
  close: number | null
  score: number | null
  fresh: SignalVerdict | null
  holding: SignalVerdict | null
  stop: number | null
  stop_basis: string | null
  target: number | null
  target2: number | null
  reward_risk: number | null
  if_breakout: IfBreakout | null
  entry_zone: [number | null, number | null]
  support: SignalLevel | null
  resistance: SignalLevel | null
  breakout: {
    above_all_resistance: boolean
    recent_breakout: { pattern: string; date: string; english: string | null } | null
  }
  factors: { name: string; score: number; detail: string }[]
  outlook_1w: { p_up: number | null; verdict: string | null; n: number | null; fallback: boolean | null }
  in_earnings_blackout: boolean
  next_earnings: string | null
  expected_results_window: { start: string; end: string; basis: string } | null
  results_overdue: boolean
  market_regime: string | null
  /** history window the analysis ran over (absent on pre-range snapshots) */
  range?: { years: number | null; start: string | null; end: string | null }
  /** when this snapshot was computed and written to the local store */
  stored_at: string
  /** present only on a live /api/signal response, not on stored snapshots */
  cached?: boolean
}

export interface SreCreds {
  clientId: string
  token: string
}

const CLIENT_KEY = 'sre.client_id'
const TOKEN_KEY = 'sre.token'

export function loadCreds(): SreCreds {
  return {
    clientId: sessionStorage.getItem(CLIENT_KEY) ?? '',
    token: sessionStorage.getItem(TOKEN_KEY) ?? '',
  }
}

export function saveCreds(creds: SreCreds) {
  sessionStorage.setItem(CLIENT_KEY, creds.clientId)
  sessionStorage.setItem(TOKEN_KEY, creds.token)
}

export function clearCreds() {
  sessionStorage.removeItem(CLIENT_KEY)
  sessionStorage.removeItem(TOKEN_KEY)
}

/** SRE creds persisted server-side per portal user, so fields come preset. */
export async function fetchServerCreds(): Promise<SreCreds | null> {
  try {
    const resp = await request<{ client_id: string | null; token: string | null }>('/api/sre/creds')
    if (!resp.client_id || !resp.token) return null
    return { clientId: resp.client_id, token: resp.token }
  } catch {
    return null
  }
}

export function saveServerCreds(creds: SreCreds): void {
  void request('/api/sre/creds', {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ client_id: creds.clientId, token: creds.token }),
  }).catch(() => undefined)
}

/** Unix expiry of an SRE bearer token, or null if undecodable. */
export function tokenExp(token: string): number | null {
  try {
    const payload = JSON.parse(atob(token.split('.')[1].replace(/-/g, '+').replace(/_/g, '/')))
    const exp = Number(payload.exp)
    return Number.isFinite(exp) ? exp : null
  } catch {
    return null
  }
}

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await authFetch(url, init)
  if (!res.ok) {
    let detail = `Request failed (${res.status})`
    try {
      const body = await res.json()
      if (body?.detail) detail = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail)
    } catch {
      // non-JSON error body; keep the generic message
    }
    const err = new Error(detail) as Error & { status?: number }
    err.status = res.status
    throw err
  }
  return res.json()
}

function post<T>(url: string, body: unknown): Promise<T> {
  return request<T>(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
}

export function fetchSreHoldings(creds: SreCreds): Promise<HoldingsResponse> {
  return post('/api/sre/holdings', { client_id: creds.clientId, token: creds.token })
}

export function fetchSreNews(creds: SreCreds, symbol: string): Promise<{ items: Record<string, unknown>[]; raw: unknown }> {
  return post('/api/sre/news', { client_id: creds.clientId, token: creds.token, symbol })
}

/** Groww demat holdings — server authenticates via TOTP keys, nothing to paste. */
export function fetchGrowwHoldings(): Promise<{ holdings: SreHolding[]; count: number }> {
  return request('/api/groww/holdings')
}

/** Compute and persist a fresh snapshot. force=true bypasses the server's
 * 15-min TTL cache so an explicit Analyse press is a guaranteed recompute. */
export function fetchSignal(
  yahooSymbol: string,
  range: RangeQuery = { years: 1 },
  force = false,
): Promise<CompactSignal> {
  const q = rangeToQuery(range)
  if (force) q.set('force', '1')
  return request(`/api/signal/${encodeURIComponent(yahooSymbol)}?${q}`)
}

/** Read previously stored snapshots only — instant, never computes. */
export async function fetchStoredSignals(symbols: string[]): Promise<Record<string, CompactSignal>> {
  if (!symbols.length) return {}
  const resp = await request<{ signals: Record<string, CompactSignal> }>(
    `/api/signals?symbols=${encodeURIComponent(symbols.join(','))}`,
  )
  return resp.signals
}

// ---- Claude holding takes (on-demand judgment layer, persisted server-side) ----

export interface ClaudeTake {
  symbol: string
  tier: 'quick' | 'deep'
  text: string
  model: string
  position: { quantity: number | null; avg_price: number | null } | null
  as_of: string | null
  stored_at: string
}

export interface TakePair {
  quick?: ClaudeTake
  deep?: ClaudeTake
}

/** Ask Claude for a take on this holding. quick = payload-only (seconds);
 * deep = adds server-side web search (can take a minute). */
export function fetchClaudeTake(
  symbol: string,
  opts: { quantity?: number | null; avg_price?: number | null; deep?: boolean },
): Promise<ClaudeTake> {
  return post('/api/claude/take', {
    symbol,
    quantity: opts.quantity ?? null,
    avg_price: opts.avg_price ?? null,
    deep: opts.deep ?? false,
  })
}

/** Read previously stored takes only — instant, never calls Claude. */
export async function fetchStoredTakes(symbols: string[]): Promise<Record<string, TakePair>> {
  if (!symbols.length) return {}
  const resp = await request<{ takes: Record<string, TakePair> }>(
    `/api/claude/takes?symbols=${encodeURIComponent(symbols.join(','))}`,
  )
  return resp.takes
}
