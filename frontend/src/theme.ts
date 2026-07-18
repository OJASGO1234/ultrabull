import { useEffect, useState } from 'react'

/** Chart color tokens, from the validated reference palette (dataviz skill).
 * Marks wear series colors; text always wears ink tokens. */
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

const light: Theme = {
  mode: 'light',
  surface: '#fcfcfb',
  page: '#f9f9f7',
  textPrimary: '#0b0b0b',
  textSecondary: '#52514e',
  muted: '#898781',
  grid: '#e1e0d9',
  baseline: '#c3c2b7',
  border: 'rgba(11,11,11,0.10)',
  blue: '#2a78d6',
  green: '#008300',
  magenta: '#e87ba4',
  yellow: '#eda100',
  aqua: '#1baf7a',
  orange: '#eb6834',
  violet: '#4a3aa7',
  red: '#e34948',
  good: '#0ca30c',
  warning: '#fab219',
  critical: '#d03b3b',
  deltaUpText: '#006300',
  deltaDownText: '#d03b3b',
  areaWash: 'rgba(42,120,214,0.10)',
  histUp: 'rgba(12,163,12,0.45)',
  histDown: 'rgba(208,59,59,0.45)',
  volUp: 'rgba(12,163,12,0.35)',
  volDown: 'rgba(208,59,59,0.35)',
}

const dark: Theme = {
  mode: 'dark',
  surface: '#1a1a19',
  page: '#0d0d0d',
  textPrimary: '#ffffff',
  textSecondary: '#c3c2b7',
  muted: '#898781',
  grid: '#2c2c2a',
  baseline: '#383835',
  border: 'rgba(255,255,255,0.10)',
  blue: '#3987e5',
  green: '#008300',
  magenta: '#d55181',
  yellow: '#c98500',
  aqua: '#199e70',
  orange: '#d95926',
  violet: '#9085e9',
  red: '#e66767',
  good: '#0ca30c',
  warning: '#fab219',
  critical: '#d03b3b',
  deltaUpText: '#0ca30c',
  deltaDownText: '#e66767',
  areaWash: 'rgba(57,135,229,0.10)',
  histUp: 'rgba(12,163,12,0.45)',
  histDown: 'rgba(208,59,59,0.45)',
  volUp: 'rgba(12,163,12,0.35)',
  volDown: 'rgba(208,59,59,0.35)',
}

export function useTheme(): Theme {
  const [mode, setMode] = useState<'light' | 'dark'>(() =>
    window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light',
  )
  useEffect(() => {
    const mq = window.matchMedia('(prefers-color-scheme: dark)')
    const onChange = (e: MediaQueryListEvent) => setMode(e.matches ? 'dark' : 'light')
    mq.addEventListener('change', onChange)
    return () => mq.removeEventListener('change', onChange)
  }, [])
  return mode === 'dark' ? dark : light
}
