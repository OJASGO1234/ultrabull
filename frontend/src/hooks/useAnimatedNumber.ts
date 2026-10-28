import { useEffect, useRef, useState } from 'react'

const prefersReducedMotion = () =>
  typeof window !== 'undefined' && window.matchMedia('(prefers-reduced-motion: reduce)').matches

/** Smoothly tweens a displayed number toward `target` when the REAL value changes.
 *
 * - First render: shows `target` immediately (no count-up from zero -- that
 *   would imply activity that never happened).
 * - Later changes: eases from what's on screen to the new value (~450ms).
 * - `prefers-reduced-motion`: jumps straight to the value.
 * - `null` / `undefined` passes through as `null` so callers can show "—".
 */
export function useAnimatedNumber(target: number | null | undefined, duration = 450): number | null {
  const goal = target ?? null
  const [display, setDisplay] = useState<number | null>(goal)
  const shown = useRef<number | null>(goal) // what is currently on screen

  useEffect(() => {
    const from = shown.current
    if (goal == null || from == null || from === goal || prefersReducedMotion()) {
      shown.current = goal
      setDisplay(goal)
      return
    }
    let raf = 0
    const t0 = performance.now()
    const step = (now: number) => {
      const k = Math.min(1, (now - t0) / duration)
      const eased = 1 - Math.pow(1 - k, 3) // ease-out cubic
      const v = from + (goal - from) * eased
      shown.current = v
      setDisplay(v)
      if (k < 1) raf = requestAnimationFrame(step)
    }
    raf = requestAnimationFrame(step)
    return () => cancelAnimationFrame(raf)
  }, [goal, duration])

  return display
}

/** True for `ms` after `value` changes (never on first render). Drives the
 * brief highlight on a metric whose real value just moved. */
export function useJustChanged(value: unknown, ms = 900): boolean {
  const prev = useRef(value)
  const [changed, setChanged] = useState(false)
  useEffect(() => {
    if (Object.is(prev.current, value)) return
    prev.current = value
    setChanged(true)
    const id = window.setTimeout(() => setChanged(false), ms)
    return () => window.clearTimeout(id)
  }, [value, ms])
  return changed
}
