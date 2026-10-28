import { useState } from 'react'
import type { FormEvent } from 'react'
import { login } from '../auth'
import { BrandMark, Button } from './ui'

export function Login({ onSuccess }: { onSuccess: (username: string) => void }) {
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const onSubmit = async (e: FormEvent) => {
    e.preventDefault()
    if (busy || !username.trim() || !password) return
    setBusy(true)
    setError(null)
    try {
      onSuccess(await login(username.trim(), password))
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="login-wrap">
      <form className="card login-card" onSubmit={onSubmit}>
        <div className="brand"><BrandMark /><h1>Ultrabull</h1></div>
        <p className="chart-note">Sign in to continue.</p>
        <input
          value={username}
          onChange={(e) => setUsername(e.target.value)}
          placeholder="Username"
          aria-label="Username"
          autoComplete="username"
          autoFocus
          spellCheck={false}
        />
        <input
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          placeholder="Password"
          aria-label="Password"
          type="password"
          autoComplete="current-password"
        />
        {error && <p className="detail-warn">{error}</p>}
        <Button type="submit" loading={busy} disabled={!username.trim() || !password}>
          {busy ? 'Signing in…' : 'Sign in'}
        </Button>
      </form>
    </div>
  )
}
