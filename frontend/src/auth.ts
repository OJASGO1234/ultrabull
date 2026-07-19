// Token plumbing for the hosted deployment. The token lives in localStorage
// and rides on every /api request; any 401 broadcasts an event that drops the
// app back to the login screen.

const TOKEN_KEY = 'ultrabull.token'
export const UNAUTHORIZED_EVENT = 'ultrabull:unauthorized'

export function getToken(): string | null {
  return localStorage.getItem(TOKEN_KEY)
}

export function setToken(token: string) {
  localStorage.setItem(TOKEN_KEY, token)
}

export function clearToken() {
  localStorage.removeItem(TOKEN_KEY)
}

export function authHeaders(): Record<string, string> {
  const token = getToken()
  return token ? { Authorization: `Bearer ${token}` } : {}
}

/** fetch that attaches the auth token and broadcasts on 401. */
export async function authFetch(url: string, init: RequestInit = {}): Promise<Response> {
  const res = await fetch(url, {
    ...init,
    headers: { ...(init.headers as Record<string, string> | undefined), ...authHeaders() },
  })
  if (res.status === 401 && !url.startsWith('/api/auth/login')) {
    clearToken()
    window.dispatchEvent(new Event(UNAUTHORIZED_EVENT))
  }
  return res
}

export interface MeResponse {
  auth_required: boolean
  username: string | null
}

export async function fetchMe(): Promise<MeResponse> {
  const res = await fetch('/api/auth/me', { headers: authHeaders() })
  if (res.status === 401) return { auth_required: true, username: null }
  if (!res.ok) throw new Error(`auth check failed (${res.status})`)
  return res.json()
}

export async function login(username: string, password: string): Promise<string> {
  const res = await fetch('/api/auth/login', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username, password }),
  })
  if (!res.ok) {
    let detail = `Login failed (${res.status})`
    try {
      const body = await res.json()
      if (body?.detail) detail = body.detail
    } catch {
      // keep generic message
    }
    throw new Error(detail)
  }
  const body = await res.json()
  setToken(body.token)
  return body.username
}
