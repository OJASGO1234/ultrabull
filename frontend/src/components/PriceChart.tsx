import { useEffect, useRef, useState } from 'react'
import {
  CandlestickSeries,
  HistogramSeries,
  LineSeries,
  createSeriesMarkers,
  type MouseEventParams,
  type SeriesMarker,
  type Time,
} from 'lightweight-charts'
import type { AnalysisResponse } from '../types'
import type { Theme } from '../theme'
import { makeChart, fmt } from './chart'
import { Legend } from './Legend'

interface Props {
  data: AnalysisResponse
  theme: Theme
}

interface Hover {
  open?: number
  high?: number
  low?: number
  close?: number
}

/** Candles + MA20/MA50 + Bollinger bounds + volume, with trade entry/exit markers. */
export function PriceChart({ data, theme: t }: Props) {
  const ref = useRef<HTMLDivElement>(null)
  const [hover, setHover] = useState<Hover>({})

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

    // Trade markers: entries carry the pattern tag, exits carry the exit reason.
    const markers: SeriesMarker<Time>[] = []
    for (const tr of data.trades) {
      if (tr.entry_date) {
        markers.push({
          time: tr.entry_date as Time,
          position: tr.side === 'long' ? 'belowBar' : 'aboveBar',
          shape: tr.side === 'long' ? 'arrowUp' : 'arrowDown',
          color: tr.side === 'long' ? t.blue : t.violet,
          text: tr.tag,
        })
      }
      if (tr.exit_date) {
        markers.push({
          time: tr.exit_date as Time,
          position: tr.side === 'long' ? 'aboveBar' : 'belowBar',
          shape: tr.side === 'long' ? 'arrowDown' : 'arrowUp',
          color: tr.side === 'long' ? t.orange : t.red,
          text: tr.reason,
        })
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
      chart.unsubscribeCrosshairMove(onMove)
      chart.remove()
    }
  }, [data, t])

  const ohlc =
    hover.close != null
      ? `O ${fmt.price(hover.open)}  H ${fmt.price(hover.high)}  L ${fmt.price(hover.low)}  C ${fmt.price(hover.close)}`
      : undefined

  return (
    <div className="card">
      <div className="card-head">
        <h2>Price — candles, MA20/MA50, Bollinger bounds</h2>
        {ohlc && <span className="hover-readout">{ohlc}</span>}
      </div>
      <Legend
        items={[
          { color: t.good, label: 'Up candle', shape: 'square' },
          { color: t.critical, label: 'Down candle', shape: 'square' },
          { color: t.orange, label: 'MA20' },
          { color: t.violet, label: 'MA50' },
          { color: t.blue, label: 'BB ±2σ' },
        ]}
      />
      <div ref={ref} className="chart" />
      <Legend
        items={[
          { color: t.blue, label: '▲ Entry long', shape: 'dot' },
          { color: t.orange, label: '▼ Exit long', shape: 'dot' },
          { color: t.violet, label: '▼ Entry short', shape: 'dot' },
          { color: t.red, label: '▲ Exit short', shape: 'dot' },
        ]}
      />
    </div>
  )
}
