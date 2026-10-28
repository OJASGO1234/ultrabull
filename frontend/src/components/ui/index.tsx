import type { ButtonHTMLAttributes, ReactNode } from 'react'
import { useAnimatedNumber, useJustChanged } from '../../hooks/useAnimatedNumber'

/* Reusable building blocks of the UltraBull design system.
   Styles live in src/styles/ui.css (+ .card/.tile in index.css). */

export type Tone = 'up' | 'down' | 'warn' | 'neutral'

/** Title row of a card: heading on the left, optional meta text + actions on the right. */
export function SectionHeader({ title, meta, actions }: { title: ReactNode; meta?: ReactNode; actions?: ReactNode }) {
  return (
    <div className="card-head">
      <h2>
        {title}
        {meta != null && <span className="section-meta">{meta}</span>}
      </h2>
      {actions && <div className="section-actions">{actions}</div>}
    </div>
  )
}

/** A card with a SectionHeader. `className` can carry layout classes like `span-6`. */
export function Panel({
  title, meta, actions, className = '', lift = false, children,
}: {
  title?: ReactNode
  meta?: ReactNode
  actions?: ReactNode
  className?: string
  lift?: boolean
  children: ReactNode
}) {
  return (
    <section className={`card panel ${lift ? 'panel-lift' : ''} ${className}`.trim()}>
      {title != null && <SectionHeader title={title} meta={meta} actions={actions} />}
      {children}
    </section>
  )
}

/** One number with a label. Numeric `value`s ease toward new real values and
 * briefly highlight when they change; pass `display` for non-numeric text. */
export function MetricCard({
  label, value, format, display, tone = 'neutral', note, noteTone, className = '',
}: {
  label: string
  value?: number | null
  format?: (n: number) => string
  display?: string
  tone?: Tone
  note?: ReactNode
  noteTone?: Tone
  className?: string
}) {
  const animated = useAnimatedNumber(value)
  const changed = useJustChanged(value)
  const text = value !== undefined
    ? (animated == null ? '—' : (format ? format(animated) : String(Math.round(animated))))
    : (display ?? '—')
  return (
    <div className={`tile metric${changed ? ' is-updated' : ''}${className ? ` ${className}` : ''}`}>
      <div className="tile-label">{label}</div>
      <div className={`tile-value tone-${tone}`}>{text}</div>
      {note != null && <div className={`tile-note tone-${noteTone ?? 'neutral'}`}>{note}</div>}
    </div>
  )
}

/** Status chip: dot + text (colour never travels alone). Set `pulse` ONLY when the
 * thing it describes is genuinely live right now. */
export function StatusPill({
  tone = 'neutral', label, pulse = false, title,
}: {
  tone?: 'good' | 'bad' | 'warn' | 'info' | 'neutral'
  label: ReactNode
  pulse?: boolean
  title?: string
}) {
  return (
    <span className={`status-pill pill-${tone}${pulse ? ' is-pulsing' : ''}`} title={title}>
      <span className="pill-dot" aria-hidden="true" />
      {label}
    </span>
  )
}

/** Button with a loading state. `ghost` and `mini` reuse the existing button classes. */
export function Button({
  variant = 'primary', loading = false, children, disabled, className = '', type = 'button', ...rest
}: ButtonHTMLAttributes<HTMLButtonElement> & { variant?: 'primary' | 'ghost' | 'mini'; loading?: boolean }) {
  const cls = variant === 'mini' ? 'btn-mini' : variant === 'ghost' ? 'btn btn-ghost' : 'btn btn-primary'
  return (
    <button type={type} className={`${cls} ${className}`.trim()} disabled={disabled || loading} aria-busy={loading || undefined} {...rest}>
      {loading && <span className="btn-spinner" aria-hidden="true" />}
      {children}
    </button>
  )
}

/** Placeholder bars while data loads. */
export function Skeleton({ lines = 3, height = 14 }: { lines?: number; height?: number }) {
  return (
    <div className="skeleton-stack" aria-busy="true" aria-label="Loading">
      {Array.from({ length: lines }, (_, i) => (
        <span key={i} className="skeleton" style={{ height, width: i === lines - 1 ? '60%' : '100%' }} />
      ))}
    </div>
  )
}

/** "Nothing here yet" box with a reason, instead of a bare grey line. */
export function EmptyState({ title, hint }: { title: string; hint?: string }) {
  return (
    <div className="empty-card">
      <strong>{title}</strong>
      {hint && <span>{hint}</span>}
    </div>
  )
}

/** Horizontal gauge: `value` against `max`. Tone shifts amber/red as it nears the limit. */
export function MeterBar({ value, max, label }: { value: number; max: number; label: string }) {
  const ratio = max > 0 ? Math.max(0, Math.min(1, value / max)) : 0
  const tone = ratio >= 1 ? 'bad' : ratio >= 0.7 ? 'warn' : 'ok'
  return (
    <div>
      <div
        className={`meter tone-${tone}`}
        role="meter"
        aria-label={label}
        aria-valuemin={0}
        aria-valuemax={max}
        aria-valuenow={Math.min(value, max)}
      >
        <div className="meter-fill" style={{ width: `${ratio * 100}%` }} />
      </div>
      <div className="meter-scale"><span>0%</span><span>limit {max}%</span></div>
    </div>
  )
}

/** Brand logo mark (inline SVG: no image file, scales cleanly). */
export function BrandMark() {
  return (
    <svg className="brand-mark" viewBox="0 0 40 40" aria-hidden="true">
      <defs>
        <linearGradient id="bm-g" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stopColor="#2dd4bf" />
          <stop offset="1" stopColor="#a3e635" />
        </linearGradient>
      </defs>
      <rect x="1" y="1" width="38" height="38" rx="11" fill="#111416" stroke="rgba(255,255,255,0.12)" />
      <path d="M8 27 L15 19 L21 24 L32 11" fill="none" stroke="url(#bm-g)" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" />
      <circle cx="32" cy="11" r="2.6" fill="#a3e635" />
    </svg>
  )
}
