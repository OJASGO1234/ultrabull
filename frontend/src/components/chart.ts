import { createChart, type IChartApi } from 'lightweight-charts'
import type { Theme } from '../theme'

/** Common chart chrome: recessive hairline grid/axes, ink text, transparent bg
 * (the surrounding card supplies the surface color).
 * The container gets a FIXED pixel height here — autoSize then only tracks
 * width. Without this, autoSize + a content-driven container height feed back
 * into each other and the chart grows downward forever. */
export function makeChart(el: HTMLElement, t: Theme, height: number): IChartApi {
  el.style.height = `${height}px`
  return createChart(el, {
    autoSize: true,
    layout: {
      background: { color: 'transparent' },
      textColor: t.muted,
      fontSize: 11,
      fontFamily: '"Segoe UI Variable Text", "Segoe UI", system-ui, -apple-system, Roboto, sans-serif',
    },
    grid: {
      vertLines: { color: t.grid },
      horzLines: { color: t.grid },
    },
    rightPriceScale: { borderColor: t.baseline },
    timeScale: { borderColor: t.baseline, timeVisible: false },
    crosshair: {
      vertLine: { color: t.muted, width: 1, labelBackgroundColor: t.textSecondary },
      horzLine: { color: t.muted, width: 1, labelBackgroundColor: t.textSecondary },
    },
  })
}

export const fmt = {
  price: (v: number | null | undefined, digits = 2) =>
    v == null ? '—' : v.toLocaleString('en-IN', { maximumFractionDigits: digits, minimumFractionDigits: digits }),
  int: (v: number | null | undefined) => (v == null ? '—' : Math.round(v).toLocaleString('en-IN')),
  pct: (v: number | null | undefined, digits = 1) =>
    v == null ? '—' : `${(v * 100).toFixed(digits)}%`,
  ratio: (v: number | 'inf' | null | undefined, digits = 2) =>
    v == null ? '—' : v === 'inf' ? '∞' : v.toFixed(digits),
  compact: (v: number | null | undefined) =>
    v == null ? '—' : Intl.NumberFormat('en-IN', { notation: 'compact', maximumFractionDigits: 1 }).format(v),
}
