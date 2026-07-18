import { useEffect, useRef, useState } from 'react'
import { LineSeries, LineStyle, type MouseEventParams, type Time } from 'lightweight-charts'
import type { AnalysisResponse } from '../types'
import type { Theme } from '../theme'
import { makeChart, fmt } from './chart'

interface Props {
  data: AnalysisResponse
  theme: Theme
}

/** RSI(14) with the conventional 30/70 reference levels (muted) and the
 * pipeline's actual 40/60 trigger levels (secondary ink). */
export function RsiChart({ data, theme: t }: Props) {
  const ref = useRef<HTMLDivElement>(null)
  const [hover, setHover] = useState<number | null>(null)

  useEffect(() => {
    const el = ref.current
    if (!el) return
    const chart = makeChart(el, t, 170)

    const rsi = chart.addSeries(LineSeries, {
      color: t.blue,
      lineWidth: 2,
      lastValueVisible: false,
      priceLineVisible: false,
    })
    rsi.setData(data.indicators.rsi.map((p) => ({ time: p.time as Time, value: p.value })))

    for (const { price, color, title } of [
      { price: 70, color: t.baseline, title: '70' },
      { price: 30, color: t.baseline, title: '30' },
      { price: 60, color: t.muted, title: 'trigger 60' },
      { price: 40, color: t.muted, title: 'trigger 40' },
    ]) {
      rsi.createPriceLine({
        price,
        color,
        lineWidth: 1,
        lineStyle: LineStyle.Solid,
        axisLabelVisible: true,
        title,
      })
    }

    const onMove = (param: MouseEventParams) => {
      const v = param.seriesData.get(rsi) as { value?: number } | undefined
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
        <h2>RSI(14)</h2>
        {hover != null && <span className="hover-readout">{fmt.price(hover, 1)}</span>}
      </div>
      <div ref={ref} className="chart" />
      {data.interpretations?.rsi && (
        <p className="chart-note chart-inference">{data.interpretations.rsi}</p>
      )}
      <p className="chart-note">
        40/60 are the pipeline's actual crossing triggers; 30/70 are visual reference only.
      </p>
    </div>
  )
}
