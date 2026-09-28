import { describe, expect, it } from 'vitest'
import { describeQueryError } from './queryError'

describe('describeQueryError', () => {
  it('names the chosen provider, not Ollama, when the server sends no reason', () => {
    const { message } = describeQueryError(500, null, 'scaleway')
    expect(message).toMatch(/Scaleway/)
    expect(message).not.toMatch(/Ollama/)
  })

  it('points to Ollama only when Ollama is the chosen provider', () => {
    expect(describeQueryError(500, null, 'ollama').message).toMatch(/Ollama/)
  })

  it("shows the server's reason as is", () => {
    const detail = 'Scaleway has no answer model on this server: set SCW_MODEL.'
    expect(describeQueryError(503, { detail }, 'scaleway').message).toBe(detail)
  })

  it('joins validation errors', () => {
    const body = { detail: [{ msg: 'field required' }, { msg: 'too long' }] }
    expect(describeQueryError(422, body, 'gemini').message).toBe('field required; too long')
  })

  it('opens Settings when the provider needs setup or a key', () => {
    expect(describeQueryError(503, { detail: 'set SCW_MODEL' }, 'scaleway').openSettings).toBe(true)
    expect(describeQueryError(400, { detail: 'Please provide a valid API key' }, 'openai').openSettings).toBe(true)
    expect(describeQueryError(429, { detail: 'API rate limit exceeded' }, 'gemini').openSettings).toBe(true)
  })

  it('keeps Settings closed when Ollama itself is down', () => {
    const body = { detail: 'Ollama is not running. Start it with: ollama serve' }
    expect(describeQueryError(503, body, 'ollama').openSettings).toBe(false)
  })
})
