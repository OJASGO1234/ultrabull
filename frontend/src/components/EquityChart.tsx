import { useEffect, useRef, useState } from 'react'
import { AreaSeries, type MouseEventParams, type Time } from 'lightweight-charts'
import type { AnalysisResponse } from '../types'
import type { Theme } from '../theme'
import { makeChart, fmt } from './chart'

interface Props {
  data: AnalysisResponse
  theme: Theme
}

/** Backtest equity curve (mark-to-market). Single series — no legend box needed. */
export function EquityChart({ data, theme: t }: Props) {
  const ref = useRef<HTMLDivElement>(null)
  const [hover, setHover] = useState<number | null>(null)

  useEffect(() => {
    const el = ref.current
    if (!el) return
    const chart = makeChart(el, t, 220)

    const area = chart.addSeries(AreaSeries, {
      lineColor: t.blue,
      lineWidth: 2,
      topColor: t.areaWash,
      bottomColor: 'transparent',
      lastValueVisible: true,
      priceLineVisible: false,
      priceFormat: { type: 'volume' },
    })
    area.setData(data.equity.map((p) => ({ time: p.time as Time, value: p.value })))

    const onMove = (param: MouseEventParams) => {
      const v = param.seriesData.get(area) as { value?: number } | undefined
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
        <h2>Equity curve (₹, mark-to-market)</h2>
        {hover != null && <span className="hover-readout">₹{fmt.int(hover)}</span>}
      </div>
      <div ref={ref} className="chart" />
    </div>
  )
}
