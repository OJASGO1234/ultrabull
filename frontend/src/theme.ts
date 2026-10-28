/** Chart/inline colour tokens (JavaScript side of the design system).
 *
 * Colours reach the screen by two routes: DOM elements read the CSS variables in
 * styles/tokens.css; canvas charts (lightweight-charts) and inline
 * `style={{ color }}` props read the values below -- CSS variables can't reach
 * into a <canvas>. Each value notes its matching CSS variable. Change both.
 *
 * The app is a single dark theme, so `useTheme()` just returns it. The hook
 * (and `Theme` type) are kept so every component's `theme` prop still works.
 * Marks wear series colours; text always wears ink tokens.
 */
export interface Theme {
  mode: 'light' | 'dark'
  surface: string
  page: string
  textPrimary: string
  textSecondary: string
  muted: string
  grid: string
  baseline: string
  border: string
  // categorical slots (fixed order, never cycled)
  blue: string
  green: string
  magenta: string
  yellow: string
  aqua: string
  orange: string
  violet: string
  red: string
  // status (reserved: polarity/state, never "series 4")
  good: string
  warning: string
  critical: string
  // delta text tokens
  deltaUpText: string
  deltaDownText: string
  // derived washes
  areaWash: string
  histUp: string
  histDown: string
  volUp: string
  volDown: string
}

export const darkTheme: Theme = {
  mode: 'dark',
  surface: '#111416', // --surface-1
  page: '#0a0c0d', // --bg
  textPrimary: '#f4f6f5', // --text-primary
  textSecondary: '#aab3b0', // --text-secondary
  muted: '#7b8683', // --muted
  grid: '#1c2124', // --grid
  baseline: '#2a3135', // --baseline
  border: 'rgba(255,255,255,0.07)', // --border
  // categorical slots -- teal leads, as the primary accent
  blue: '#2dd4bf', // teal (--accent): primary series, BUY labels
  green: '#a3e635', // lime (--good)
  magenta: '#e879a9',
  yellow: '#fbbf24', // amber (--warning)
  aqua: '#60a5fa', // sky blue: kept distinct from teal
  orange: '#fb923c', // SELL labels
  violet: '#a78bfa',
  red: '#f87171', // coral (--critical)
  // status
  good: '#a3e635',
  warning: '#fbbf24',
  critical: '#f87171',
  // delta text: gains lime, losses coral
  deltaUpText: '#a3e635',
  deltaDownText: '#f87171',
  // derived washes (translucent versions of the accent / status colours)
  areaWash: 'rgba(45,212,191,0.14)',
  histUp: 'rgba(163,230,53,0.45)',
  histDown: 'rgba(248,113,113,0.45)',
  volUp: 'rgba(163,230,53,0.35)',
  volDown: 'rgba(248,113,113,0.35)',
}

export function useTheme(): Theme {
  return darkTheme
}
