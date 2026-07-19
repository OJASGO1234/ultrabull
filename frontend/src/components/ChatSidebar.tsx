import { useEffect, useRef, useState } from 'react'
import type { FormEvent } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import type { RangeQuery } from '../api'
import { authFetch } from '../auth'

interface ChatMsg {
  role: 'user' | 'assistant'
  content: string
}

interface Props {
  symbol: string
  range: RangeQuery
  hasData: boolean
}

/** Toggleable analysis-assistant panel: Claude answers questions grounded in
 * the page's current analysis payload (server injects it as context). */
export function ChatSidebar({ symbol, range, hasData }: Props) {
  const [open, setOpen] = useState(false)
  const [messages, setMessages] = useState<ChatMsg[]>([])
  const [input, setInput] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const scrollRef = useRef<HTMLDivElement>(null)

  // New symbol/range = new conversation context
  useEffect(() => {
    setMessages([])
    setError(null)
  }, [symbol, range.years, range.start, range.end])

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight })
  }, [messages, busy])

  const send = async (e: FormEvent) => {
    e.preventDefault()
    const text = input.trim()
    if (!text || busy || !hasData) return
    const next: ChatMsg[] = [...messages, { role: 'user', content: text }]
    setMessages(next)
    setInput('')
    setBusy(true)
    setError(null)
    try {
      const res = await authFetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          symbol,
          years: range.years ?? 0.5,
          start: range.start ?? null,
          end: range.end ?? null,
          messages: next,
        }),
      })
      if (!res.ok) {
        let detail = `Request failed (${res.status})`
        try {
          const body = await res.json()
          if (body?.detail) detail = body.detail
        } catch { /* keep generic */ }
        throw new Error(detail)
      }
      const body = await res.json()
      setMessages([...next, { role: 'assistant', content: body.reply }])
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
      setMessages(messages) // roll back the optimistic user message
      setInput(text)
    } finally {
      setBusy(false)
    }
  }

  return (
    <>
      {!open && (
        <button
          type="button"
          className="chat-fab"
          onClick={() => setOpen(true)}
          aria-label="Open analysis chat"
        >
          Ask
        </button>
      )}

      {open && (
        <aside className="chat-panel" aria-label="Analysis assistant">
          <div className="chat-head">
            <div className="chat-head-text">
              <b>Analysis assistant</b>
              <span className="chat-sub">
                {hasData ? `grounded in the ${symbol} view` : 'run an analysis first'}
              </span>
            </div>
            <button
              type="button"
              className="chat-close"
              onClick={() => setOpen(false)}
              aria-label="Close analysis chat"
            >
              ✕
            </button>
          </div>

          <div className="chat-messages" ref={scrollRef}>
            {messages.length === 0 && !busy && (
              <div className="chat-hint">
                Ask about what's on the page — e.g. "Why is the verdict WAIT?",
                "What would invalidate the long thesis?" — or have it <b>backtest an
                idea</b>: "test if RSI&lt;30 plus a 2× volume spike predicts bounces".
                Rule tests run against the 4-year, 35-symbol NSE study harness and
                report t-stats plus a recent-12-months stability check.
                Answers use this app's own validated studies; it analyzes, you decide.
              </div>
            )}
            {messages.map((m, i) =>
              m.role === 'assistant' ? (
                <div key={i} className="chat-msg chat-assistant chat-md">
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>{m.content}</ReactMarkdown>
                </div>
              ) : (
                <div key={i} className="chat-msg chat-user">
                  {m.content}
                </div>
              ),
            )}
            {busy && <div className="chat-msg chat-assistant chat-busy">Thinking…</div>}
            {error && <div className="chat-error">⚠ {error}</div>}
          </div>

          <form className="chat-input" onSubmit={send}>
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault()
                  e.currentTarget.form?.requestSubmit()
                }
              }}
              placeholder={hasData ? 'Ask about this analysis…' : 'Run an analysis first'}
              rows={2}
              disabled={!hasData || busy}
            />
            <button type="submit" disabled={!hasData || busy || !input.trim()}>
              Send
            </button>
          </form>
        </aside>
      )}
    </>
  )
}
