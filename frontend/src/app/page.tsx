'use client';

import React, { useState } from 'react';
import { useRouter } from 'next/navigation';
import { useLanguage } from '@/hooks/useLanguage';
import { Language } from '@/types/consent';
import { setConsentState } from '@/lib/storage';

export default function Home() {
  const router = useRouter();
  const { language, setLanguage, t } = useLanguage();
  const [name, setName] = useState('');
  const [error, setError] = useState('');

  const handleStart = (e: React.FormEvent) => {
    e.preventDefault();
    if (!name.trim()) {
      setError(t.home.errorName);
      return;
    }

    // Store name and language in consent state
    setConsentState({
      language,
      name: name.trim(),
      currentStep: 0,
      consentsGiven: Array(6).fill(false),
      timestamp: new Date().toISOString(),
    });

    // Navigate to consent flow
    router.push('/consent');
  };

  return (
    <main className="flex min-h-screen flex-col items-center justify-center p-8 bg-gradient-to-b from-gray-50 to-gray-100">
      <div className="max-w-md w-full bg-white rounded-lg shadow-lg p-8">
        <h1 className="text-2xl font-bold text-center mb-6 text-gray-800">
          {t.home.title}
        </h1>

        <p className="text-gray-600 mb-6 text-sm">
          {t.home.description}
        </p>

        <form onSubmit={handleStart} className="space-y-4">
          {/* Language Selection */}
          <div>
            <label htmlFor="language" className="block text-sm font-medium text-gray-700 mb-1">
              {t.common.language}
            </label>
            <select
              id="language"
              value={language}
              onChange={(e) => setLanguage(e.target.value as Language)}
              className="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none transition bg-white"
            >
              <option value="en">English</option>
              <option value="ko">한국어 (Korean)</option>
            </select>
          </div>

          {/* Name Input */}
          <div>
            <label htmlFor="name" className="block text-sm font-medium text-gray-700 mb-1">
              {t.home.nameLabel}
            </label>
            <input
              type="text"
              id="name"
              value={name}
              onChange={(e) => setName(e.target.value)}
              className="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none transition"
              placeholder={t.home.namePlaceholder}
            />
          </div>

          {error && (
            <p className="text-red-500 text-sm">{error}</p>
          )}

          <button
            type="submit"
            className="w-full bg-blue-600 text-white py-2 px-4 rounded-md hover:bg-blue-700 transition disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {t.home.continueButton}
          </button>
        </form>

        <p className="mt-6 text-xs text-gray-500 text-center">
          {t.home.privacyNote}
        </p>
      </div>
    </main>
  );
}
