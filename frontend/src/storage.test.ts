import { beforeEach, describe, expect, it, vi } from 'vitest'
import { loadEnforcePrivacyMode, saveEnforcePrivacyMode } from './storage'

// Mock localStorage for Node environment
const localStorageMock = (() => {
  let store: Record<string, string> = {}
  return {
    getItem: (key: string) => store[key] || null,
    setItem: (key: string, value: string) => {
      store[key] = String(value)
    },
    removeItem: (key: string) => {
      delete store[key]
    },
    clear: () => {
      store = {}
    },
  }
})()

Object.defineProperty(window, 'localStorage', {
  value: localStorageMock,
})

describe('Privacy Mode Storage', () => {
  beforeEach(() => {
    localStorageMock.clear()
    vi.clearAllMocks()
  })

  describe('loadEnforcePrivacyMode', () => {
    it('returns false when localStorage is empty', () => {
      const result = loadEnforcePrivacyMode()
      expect(result).toBe(false)
    })

    it('returns true when localStorage has enforcePrivacyMode=true', () => {
      localStorageMock.setItem('radiationsafety_enforce_privacy_mode', 'true')
      const result = loadEnforcePrivacyMode()
      expect(result).toBe(true)
    })

    it('returns false when localStorage has enforcePrivacyMode=false', () => {
      localStorageMock.setItem('radiationsafety_enforce_privacy_mode', 'false')
      const result = loadEnforcePrivacyMode()
      expect(result).toBe(false)
    })

    it('returns false when localStorage is corrupted', () => {
      localStorageMock.setItem('radiationsafety_enforce_privacy_mode', 'invalid')
      const result = loadEnforcePrivacyMode()
      expect(result).toBe(false)
    })

    it('returns false on localStorage access error', () => {
      const getItemSpy = vi.spyOn(localStorageMock, 'getItem').mockImplementation(() => {
        throw new Error('localStorage error')
      })
      const result = loadEnforcePrivacyMode()
      expect(result).toBe(false)
      getItemSpy.mockRestore()
    })
  })

  describe('saveEnforcePrivacyMode', () => {
    it('saves true to localStorage', () => {
      saveEnforcePrivacyMode(true)
      const stored = localStorageMock.getItem('radiationsafety_enforce_privacy_mode')
      expect(stored).toBe('true')
    })

    it('saves false to localStorage', () => {
      saveEnforcePrivacyMode(false)
      const stored = localStorageMock.getItem('radiationsafety_enforce_privacy_mode')
      expect(stored).toBe('false')
    })

    it('does not throw when localStorage is unavailable', () => {
      const setItemSpy = vi.spyOn(localStorageMock, 'setItem').mockImplementation(() => {
        throw new Error('localStorage error')
      })
      expect(() => saveEnforcePrivacyMode(true)).not.toThrow()
      setItemSpy.mockRestore()
    })
  })
})
