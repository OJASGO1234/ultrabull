/** Identity channel for multi-series charts: swatch + label in ink tokens.
 * The colored mark beside the text carries identity — text never wears the series color. */
export interface LegendItem {
  color: string
  label: string
  value?: string
  shape?: 'line' | 'dot' | 'square'
}

export function Legend({ items }: { items: LegendItem[] }) {
  return (
    <div className="legend">
      {items.map((it) => (
        <span className="legend-item" key={it.label}>
          <span
            className={`legend-swatch legend-swatch-${it.shape ?? 'line'}`}
            style={{ backgroundColor: it.color }}
          />
          <span className="legend-label">{it.label}</span>
          {it.value != null && <span className="legend-value">{it.value}</span>}
        </span>
      ))}
    </div>
  )
}
