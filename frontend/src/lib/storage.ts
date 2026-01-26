/**
 * SessionStorage utilities for managing interview state
 */

import { Language, ConsentState, SessionData } from '@/types/consent';

export const STORAGE_KEYS = {
  CONSENT_STATE: 'consentState',
  LANGUAGE_PREFERENCE: 'languagePreference',
  INTERVIEW_SESSION: 'interviewSession',
} as const;

/**
 * Get language preference from sessionStorage
 */
export function getLanguage(): Language {
  if (typeof window === 'undefined') return 'en';

  const stored = sessionStorage.getItem(STORAGE_KEYS.LANGUAGE_PREFERENCE);
  if (stored === 'en' || stored === 'ko') {
    return stored;
  }
  return 'en';
}

/**
 * Set language preference in sessionStorage
 */
export function setLanguage(language: Language): void {
  if (typeof window === 'undefined') return;

  sessionStorage.setItem(STORAGE_KEYS.LANGUAGE_PREFERENCE, language);
}

/**
 * Get consent state from sessionStorage
 */
export function getConsentState(): ConsentState | null {
  if (typeof window === 'undefined') return null;

  const stored = sessionStorage.getItem(STORAGE_KEYS.CONSENT_STATE);
  if (!stored) return null;

  try {
    return JSON.parse(stored) as ConsentState;
  } catch {
    return null;
  }
}

/**
 * Set consent state in sessionStorage
 */
export function setConsentState(state: ConsentState): void {
  if (typeof window === 'undefined') return;

  sessionStorage.setItem(STORAGE_KEYS.CONSENT_STATE, JSON.stringify(state));
}

/**
 * Clear consent state from sessionStorage
 */
export function clearConsentState(): void {
  if (typeof window === 'undefined') return;

  sessionStorage.removeItem(STORAGE_KEYS.CONSENT_STATE);
}

/**
 * Get session data for passing to consent flow
 */
export function getSessionData(): SessionData | null {
  const state = getConsentState();
  if (!state) return null;

  return {
    name: state.name,
    language: state.language,
    consentTimestamp: state.timestamp || new Date().toISOString(),
  };
}
