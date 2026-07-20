// Paper-trading account client.
import { authFetch } from './auth'

export interface PaperPosition {
  symbol: string
  yahoo_symbol: string | null
  qty: number
  avg_cost: number
  cost_basis: number
  ltp: number | null
  market_value: number
  unrealized_pnl: number | null
  unrealized_pct: number | null
  opened_at: string
  stop: number | null
  target: number | null
  stop_hit: boolean
  target_hit: boolean
}

export interface PaperState {
  config: {
    starting_capital: number
    brokerage_pct: number
    tax_pct: number
    kill_drawdown_pct: number
    created_at: string
  }
  cash: number
  holdings_value: number
  equity: number
  starting_capital: number
  total_pnl: number
  total_return_pct: number | null
  realized_pnl: number
  unrealized_pnl: number
  positions: PaperPosition[]
  market_open: boolean
  executor: { running: boolean; pending_orders: number }
  stats: {
    closed_trades: number
    win_rate: number | null
    avg_win: number | null
    avg_loss: number | null
    max_drawdown_pct: number
    current_drawdown_pct: number
    kill_breached: boolean
  }
}

export interface PaperHistory {
  equity_curve: { date: string; equity: number }[]
  trades: {
    ts: string
    symbol: string
    side: string
    qty: number
    price: number
    charges: number
    realized_pnl: number | null
    note: string | null
  }[]
}

export function fetchPaper(): Promise<PaperState> {
  return authFetch('/api/paper').then((r) => {
    if (!r.ok) throw new Error(`paper fetch failed (${r.status})`)
    return r.json()
  })
}

export function fetchPaperHistory(): Promise<PaperHistory> {
  return authFetch('/api/paper/history').then((r) => r.json())
}

export function setPaperConfig(cfg: {
  starting_capital?: number
  brokerage_pct?: number
  tax_pct?: number
  kill_drawdown_pct?: number
}): Promise<{ ok: boolean; capital_locked: boolean }> {
  return authFetch('/api/paper/config', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(cfg),
  }).then((r) => r.json())
}

export interface TradeInput {
  symbol: string
  yahoo_symbol?: string | null
  side: 'BUY' | 'SELL'
  qty: number
  price?: number | null
  stop?: number | null
  target?: number | null
  note?: string | null
}

export async function paperTrade(t: TradeInput): Promise<{ ok: boolean; price: number; charges: number; realized_pnl: number | null; state: PaperState }> {
  const res = await authFetch('/api/paper/trade', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(t),
  })
  if (!res.ok) {
    let detail = `trade failed (${res.status})`
    try {
      const b = await res.json()
      if (b?.detail) detail = b.detail
    } catch {
      // keep generic
    }
    throw new Error(detail)
  }
  return res.json()
}

export function resetPaper(): Promise<{ ok: boolean }> {
  return authFetch('/api/paper/reset', { method: 'POST' }).then((r) => r.json())
}

// ---- pending (limit / stop) orders, filled automatically by the executor ----

export interface PendingOrder {
  id: number
  symbol: string
  yahoo_symbol: string
  side: string
  qty: number
  limit_price: number
  trigger: 'lte' | 'gte'
  stop: number | null
  target: number | null
  status: string
  created_at: string
}

export async function placeOrder(o: {
  symbol: string
  yahoo_symbol: string
  side: 'BUY' | 'SELL'
  qty: number
  limit_price: number
  stop?: number | null
  target?: number | null
  note?: string | null
}): Promise<{ ok: boolean; id: number; trigger: string; kind: string; current_price: number | null }> {
  const res = await authFetch('/api/paper/order', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(o),
  })
  if (!res.ok) {
    let detail = `order failed (${res.status})`
    try {
      const b = await res.json()
      if (b?.detail) detail = b.detail
    } catch {
      // keep generic
    }
    throw new Error(detail)
  }
  return res.json()
}

export function listOrders(): Promise<{ orders: PendingOrder[] }> {
  return authFetch('/api/paper/orders').then((r) => r.json())
}

export function cancelOrder(id: number): Promise<{ ok: boolean }> {
  return authFetch(`/api/paper/order/${id}/cancel`, { method: 'POST' }).then((r) => r.json())
}
