import { useEffect, useRef, useState } from 'react'
import {
  CandlestickSeries,
  HistogramSeries,
  LineSeries,
  LineStyle,
  createSeriesMarkers,
  type ISeriesApi,
  type MouseEventParams,
  type SeriesMarker,
  type Time,
} from 'lightweight-charts'
import type { AnalysisResponse } from '../types'
import type { Theme } from '../theme'
import type { LiveQuote } from '../live'
import { makeChart, fmt } from './chart'
import { Legend, type LegendItem } from './Legend'

interface Props {
  data: AnalysisResponse
  theme: Theme
  live?: LiveQuote | null
}

interface Hover {
  open?: number
  high?: number
  low?: number
  close?: number
}

interface Layers {
  indicators: boolean
  levels: boolean
  detections: boolean
  trades: boolean
  labels: boolean
  events: boolean
  indexes: boolean
}

const TOGGLES: { key: keyof Layers; label: string }[] = [
  { key: 'indicators', label: 'MA / BB' },
  { key: 'levels', label: 'S/R levels' },
  { key: 'detections', label: 'Signals' },
  { key: 'trades', label: 'Trades' },
  { key: 'labels', label: 'Labels' },
  { key: 'events', label: 'Events' },
  { key: 'indexes', label: 'Index overlay' },
]

/** Candles + toggleable layers: MA/BB, S/R levels, trade markers (+text
 * labels), corporate events, index shape overlays. Long windows default the
 * text labels off — they are most of the visual noise. */
export function PriceChart({ data, theme: t, live }: Props) {
  const ref = useRef<HTMLDivElement>(null)
  const candlesRef = useRef<ISeriesApi<'Candlestick'> | null>(null)
  const [hover, setHover] = useState<Hover>({})
  const [layers, setLayers] = useState<Layers>(() => ({
    indicators: true,
    levels: true,
    detections: true,
    trades: true,
    labels: data.bars <= 300,
    events: data.bars <= 600,
    indexes: true,
  }))

  useEffect(() => {
    const el = ref.current
    if (!el) return
    const chart = makeChart(el, t, 420)

    const candles = chart.addSeries(CandlestickSeries, {
      upColor: t.good,
      downColor: t.critical,
      wickUpColor: t.good,
      wickDownColor: t.critical,
      borderVisible: false,
    })
    candles.setData(data.ohlcv.map(({ time, open, high, low, close }) => ({
      time: time as Time, open, high, low, close,
    })))
    candlesRef.current = candles

    const volume = chart.addSeries(HistogramSeries, {
      priceScaleId: 'volume',
      priceFormat: { type: 'volume' },
      lastValueVisible: false,
      priceLineVisible: false,
    })
    volume.priceScale().applyOptions({ scaleMargins: { top: 0.82, bottom: 0 } })
    volume.setData(data.ohlcv.map((r) => ({
      time: r.time as Time,
      value: r.volume,
      color: r.close >= r.open ? t.volUp : t.volDown,
    })))

    if (layers.indicators) {
      const lineOpts = { lineWidth: 2, lastValueVisible: false, priceLineVisible: false } as const
      const ma20 = chart.addSeries(LineSeries, { color: t.orange, ...lineOpts })
      ma20.setData(data.indicators.ma20.map((p) => ({ time: p.time as Time, value: p.value })))
      const ma50 = chart.addSeries(LineSeries, { color: t.violet, ...lineOpts })
      ma50.setData(data.indicators.ma50.map((p) => ({ time: p.time as Time, value: p.value })))

      const bbOpts = { lineWidth: 1, lastValueVisible: false, priceLineVisible: false, color: t.blue } as const
      const bbU = chart.addSeries(LineSeries, bbOpts)
      bbU.setData(data.indicators.bb_upper.map((p) => ({ time: p.time as Time, value: p.value })))
      const bbL = chart.addSeries(LineSeries, bbOpts)
      bbL.setData(data.indicators.bb_lower.map((p) => ({ time: p.time as Time, value: p.value })))
    }

    // Index overlays on their own hidden scale: shape and timing vs the stock
    // without distorting the price axis (quantified comparison is in RS tiles).
    if (layers.indexes) {
      const indexColors = [t.muted, t.magenta]
      for (const [i, idx] of (data.indexes ?? []).entries()) {
        const series = chart.addSeries(LineSeries, {
          priceScaleId: 'indexes',
          color: indexColors[i % indexColors.length],
          lineWidth: 1,
          lineStyle: LineStyle.Dotted,
          lastValueVisible: false,
          priceLineVisible: false,
          crosshairMarkerVisible: false,
        })
        const first = idx.points[0]?.value
        if (!first) continue
        series.setData(idx.points.map((p) => ({
          time: p.time as Time,
          value: (p.value / first) * 100,
        })))
      }
      if (data.indexes?.length) {
        chart.priceScale('indexes').applyOptions({ scaleMargins: { top: 0.08, bottom: 0.3 } })
      }
    }

    // Support/resistance zones as horizontal dashed lines with touch counts.
    if (layers.levels) {
      for (const lv of data.levels ?? []) {
        candles.createPriceLine({
          price: lv.price,
          color: lv.kind === 'support' ? t.good : t.critical,
          lineWidth: 1,
          lineStyle: LineStyle.Dashed,
          axisLabelVisible: true,
          title: `${lv.kind === 'support' ? 'S' : 'R'}${lv.source === '52w' ? ' 52w' : ` ×${lv.touches}`}`,
        })
      }
    }

    const markers: SeriesMarker<Time>[] = []

    // Corporate events: earnings / dividends / splits, above the bar.
    if (layers.events) {
      const eventStyle: Record<string, { color: string; short: string }> = {
        EARNINGS: { color: t.yellow, short: 'E' },
        DIVIDEND: { color: t.aqua, short: 'D' },
        SPLIT: { color: t.magenta, short: 'S' },
      }
      const barTimes = new Set(data.ohlcv.map((r) => r.time))
      for (const ev of data.events ?? []) {
        const style = eventStyle[ev.type]
        if (!style || !barTimes.has(ev.date)) continue
        markers.push({
          time: ev.date as Time,
          position: 'aboveBar',
          shape: 'circle',
          color: style.color,
          text: layers.labels && ev.type === 'DIVIDEND' ? ev.label : style.short,
        })
      }
    }

    // Qualified detections (the buy/sell signal sightings): squares under/over
    // the bar. Days where the backtest actually entered a trade are skipped
    // when the trade layer is on — the entry arrow already marks those.
    if (layers.detections) {
      const barTimes = new Set(data.ohlcv.map((r) => r.time))
      const tradeEntryDates = new Set(data.trades.map((tr) => tr.entry_date).filter(Boolean))
      for (const d of data.detections) {
        if (!d.qualified || !barTimes.has(d.date)) continue
        if (layers.trades && tradeEntryDates.has(d.date)) continue
        const bullish = d.direction !== 'bearish'
        markers.push({
          time: d.date as Time,
          position: bullish ? 'belowBar' : 'aboveBar',
          shape: 'square',
          color: bullish ? t.good : t.critical,
          text: layers.labels ? d.pattern : undefined,
        })
      }
    }

    // Trade markers: entries carry the pattern tag, exits the exit reason.
    if (layers.trades) {
      for (const tr of data.trades) {
        if (tr.entry_date) {
          markers.push({
            time: tr.entry_date as Time,
            position: tr.side === 'long' ? 'belowBar' : 'aboveBar',
            shape: tr.side === 'long' ? 'arrowUp' : 'arrowDown',
            color: tr.side === 'long' ? t.blue : t.violet,
            text: layers.labels ? tr.tag : undefined,
          })
        }
        if (tr.exit_date) {
          markers.push({
            time: tr.exit_date as Time,
            position: tr.side === 'long' ? 'aboveBar' : 'belowBar',
            shape: tr.side === 'long' ? 'arrowDown' : 'arrowUp',
            color: tr.side === 'long' ? t.orange : t.red,
            text: layers.labels ? tr.reason : undefined,
          })
        }
      }
    }
    markers.sort((a, b) => String(a.time).localeCompare(String(b.time)))
    createSeriesMarkers(candles, markers)

    const onMove = (param: MouseEventParams) => {
      const bar = param.seriesData.get(candles) as Hover | undefined
      setHover(bar ?? {})
    }
    chart.subscribeCrosshairMove(onMove)
    chart.timeScale().fitContent()

    return () => {
      candlesRef.current = null
      chart.unsubscribeCrosshairMove(onMove)
      chart.remove()
    }
  }, [data, t, layers])

  // Stream live ticks into the last candle without rebuilding the chart.
  useEffect(() => {
    const candles = candlesRef.current
    if (!candles || !live || !data.ohlcv.length) return
    const today = new Intl.DateTimeFormat('en-CA', { timeZone: 'Asia/Kolkata' }).format(
      new Date(live.ts * 1000),
    )
    const last = data.ohlcv[data.ohlcv.length - 1]
    if (last.time === today) {
      candles.update({
        time: today as Time,
        open: last.open,
        high: Math.max(last.high, live.high ?? live.ltp, live.ltp),
        low: Math.min(last.low, live.low ?? live.ltp, live.ltp),
        close: live.ltp,
      })
    } else if (today > last.time) {
      candles.update({
        time: today as Time,
        open: live.open ?? live.ltp,
        high: live.high ?? live.ltp,
        low: live.low ?? live.ltp,
        close: live.ltp,
      })
    }
  }, [live, data])

  const ohlc =
    hover.close != null
      ? `O ${fmt.price(hover.open)}  H ${fmt.price(hover.high)}  L ${fmt.price(hover.low)}  C ${fmt.price(hover.close)}`
      : undefined

  const legendTop: LegendItem[] = [
    { color: t.good, label: 'Up candle', shape: 'square' },
    { color: t.critical, label: 'Down candle', shape: 'square' },
    ...(layers.indicators
      ? [
          { color: t.orange, label: 'MA20' },
          { color: t.violet, label: 'MA50' },
          { color: t.blue, label: 'BB ±2σ' },
        ]
      : []),
    ...(layers.levels
      ? [
          { color: t.good, label: 'Support (×touches)', shape: 'line' as const },
          { color: t.critical, label: 'Resistance (×touches)', shape: 'line' as const },
        ]
      : []),
    ...(layers.indexes
      ? (data.indexes ?? []).map((idx, i) => ({
          color: [t.muted, t.magenta][i % 2],
          label: `${idx.name} (shape overlay)`,
          shape: 'line' as const,
        }))
      : []),
  ]

  const legendBottom: LegendItem[] = [
    ...(layers.detections
      ? [
          { color: t.good, label: '■ Bullish signal (qualified)', shape: 'square' as const },
          { color: t.critical, label: '■ Bearish signal (qualified)', shape: 'square' as const },
        ]
      : []),
    ...(layers.trades
      ? [
          { color: t.blue, label: '▲ Entry long', shape: 'dot' as const },
          { color: t.orange, label: '▼ Exit long', shape: 'dot' as const },
          { color: t.violet, label: '▼ Entry short', shape: 'dot' as const },
          { color: t.red, label: '▲ Exit short', shape: 'dot' as const },
        ]
      : []),
    ...(layers.events
      ? [
          { color: t.yellow, label: '● Earnings', shape: 'dot' as const },
          { color: t.aqua, label: '● Dividend', shape: 'dot' as const },
          { color: t.magenta, label: '● Split/bonus', shape: 'dot' as const },
        ]
      : []),
  ]

  return (
    <div className="card">
      <div className="card-head">
        <h2>Price — candles &amp; layers</h2>
        <span className="hover-readout">
          {live && (
            <span
              className="live-readout"
              style={{ color: (live.day_change_pct ?? 0) >= 0 ? t.deltaUpText : t.deltaDownText }}
            >
              <span className="live-dot" /> ₹{fmt.price(live.ltp)}
              {live.day_change_pct != null &&
                ` ${live.day_change_pct >= 0 ? '+' : ''}${live.day_change_pct.toFixed(2)}%`}
            </span>
          )}
          {live && ohlc && ' · '}
          {ohlc}
        </span>
      </div>
      <div className="chip-row">
        {TOGGLES.map((tg) => (
          <button
            key={tg.key}
            type="button"
            className={`chip${layers[tg.key] ? ' chip-on' : ''}`}
            aria-pressed={layers[tg.key]}
            onClick={() => setLayers((l) => ({ ...l, [tg.key]: !l[tg.key] }))}
          >
            {tg.label}
          </button>
        ))}
      </div>
      <Legend items={legendTop} />
      <div ref={ref} className="chart" />
      {legendBottom.length > 0 && <Legend items={legendBottom} />}
    </div>
  )
}
