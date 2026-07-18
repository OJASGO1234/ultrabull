import { useEffect, useRef, useState } from 'react'
import { HistogramSeries, LineSeries, type MouseEventParams, type Time } from 'lightweight-charts'
import type { AnalysisResponse } from '../types'
import type { Theme } from '../theme'
import { makeChart, fmt } from './chart'
import { Legend } from './Legend'

interface Props {
  data: AnalysisResponse
  theme: Theme
}

/** MACD line + signal line + polarity histogram (SMA-based variant — see docs). */
export function MacdChart({ data, theme: t }: Props) {
  const ref = useRef<HTMLDivElement>(null)
  const [hover, setHover] = useState<number | null>(null)

  useEffect(() => {
    const el = ref.current
    if (!el) return
    const chart = makeChart(el, t, 200)

    const hist = chart.addSeries(HistogramSeries, {
      lastValueVisible: false,
      priceLineVisible: false,
    })
    hist.setData(data.indicators.macd_hist.map((p) => ({
      time: p.time as Time,
      value: p.value,
      color: p.value >= 0 ? t.histUp : t.histDown,
    })))

    const lineOpts = { lineWidth: 2, lastValueVisible: false, priceLineVisible: false } as const
    const macd = chart.addSeries(LineSeries, { color: t.blue, ...lineOpts })
    macd.setData(data.indicators.macd.map((p) => ({ time: p.time as Time, value: p.value })))
    const signal = chart.addSeries(LineSeries, { color: t.orange, ...lineOpts })
    signal.setData(data.indicators.macd_signal.map((p) => ({ time: p.time as Time, value: p.value })))

    const onMove = (param: MouseEventParams) => {
      const v = param.seriesData.get(macd) as { value?: number } | undefined
      setHover(v?.value ?? null)
    }
    chart.subscribeCrosshairMove(onMove)
    chart.timeScale().fitContent()
    return () => {
      chart.unsubscribeCrosshairMove(onMove)
      chart.remove()
    }
  }, [data, t])

  return (
    <div className="card">
      <div className="card-head">
        <h2>MACD(12, 26, 9)</h2>
        {hover != null && <span className="hover-readout">{fmt.price(hover)}</span>}
      </div>
      <Legend
        items={[
          { color: t.blue, label: 'MACD' },
          { color: t.orange, label: 'Signal' },
          { color: t.good, label: 'Hist +', shape: 'square' },
          { color: t.critical, label: 'Hist −', shape: 'square' },
        ]}
      />
      <div ref={ref} className="chart" />
      {data.interpretations?.macd && (
        <p className="chart-note chart-inference">{data.interpretations.macd}</p>
      )}
    </div>
  )
}
