'use client';

import { useState, useEffect, useRef } from 'react';
import { useRouter } from 'next/navigation';

interface Message {
  id: string;
  type: 'system' | 'user' | 'instruction';
  content: string;
  timestamp: Date;
}

interface Progress {
  current: number;
  total: number;
  phase: string;
  predefined_complete: number;
  predefined_total: number;
  main_complete: number;
  main_total: number;
  repeat_complete: number;
  repeat_total: number;
}

interface SessionData {
  sessionId: string;
  instruction: string;
  currentQuestion: string;
  phase: string;
  progress: Progress;
  name: string;
}

export default function InterviewPage() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [sessionData, setSessionData] = useState<SessionData | null>(null);
  const [isComplete, setIsComplete] = useState(false);
  const [progress, setProgress] = useState<Progress | null>(null);
  const [isAwaitingConfirmation, setIsAwaitingConfirmation] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const router = useRouter();

  useEffect(() => {
    // Load session from sessionStorage
    const stored = sessionStorage.getItem('interviewSession');
    if (!stored) {
      router.push('/');
      return;
    }

    const data: SessionData = JSON.parse(stored);
    setSessionData(data);
    setProgress(data.progress);

    // Initialize messages with instruction and first question
    setMessages([
      {
        id: 'instruction',
        type: 'instruction',
        content: data.instruction,
        timestamp: new Date(),
      },
      {
        id: 'q-0',
        type: 'system',
        content: data.currentQuestion,
        timestamp: new Date(),
      },
    ]);
  }, [router]);

  useEffect(() => {
    // Scroll to bottom when messages change
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading || !sessionData) return;

    const userMessage = input.trim();
    setInput('');
    setIsLoading(true);

    // Add user message
    const userMsgId = `user-${Date.now()}`;
    setMessages((prev) => [
      ...prev,
      {
        id: userMsgId,
        type: 'user',
        content: userMessage,
        timestamp: new Date(),
      },
    ]);

    try {
      const response = await fetch('/api/respond', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: sessionData.sessionId,
          response: userMessage,
          is_confirmation: isAwaitingConfirmation,
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to submit response');
      }

      const data = await response.json();
      setProgress(data.progress);

      if (data.is_complete) {
        setIsComplete(true);
        setIsAwaitingConfirmation(false);
        setMessages((prev) => [
          ...prev,
          {
            id: 'complete',
            type: 'system',
            content: '🎉 Thank you for completing the interview! Please wait while we evaluate your responses. This may take 1-2 minutes...',
            timestamp: new Date(),
          },
        ]);
        
        // Save results to backend - this can take 1-2 minutes for evaluation
        try {
          // Use AbortController with a longer timeout (3 minutes)
          const controller = new AbortController();
          const timeoutId = setTimeout(() => controller.abort(), 180000); // 3 minute timeout
          
          const resultsResponse = await fetch(`/api/results/${sessionData.sessionId}`, {
            signal: controller.signal,
          });
          clearTimeout(timeoutId);
          
          if (resultsResponse.ok) {
            setMessages((prev) => [
              ...prev,
              {
                id: 'saved',
                type: 'system',
                content: '✅ Your responses have been saved and evaluated successfully. You may now close this page.',
                timestamp: new Date(),
              },
            ]);
          } else {
            const errorData = await resultsResponse.json().catch(() => ({}));
            console.error('Failed to save results:', errorData);
            setMessages((prev) => [
              ...prev,
              {
                id: 'save-error',
                type: 'system',
                content: '⚠️ There was an issue saving your responses. Please contact the administrator.',
                timestamp: new Date(),
              },
            ]);
          }
        } catch (saveErr) {
          console.error('Error saving results:', saveErr);
          const errorMessage = saveErr instanceof Error && saveErr.name === 'AbortError'
            ? '⏱️ The evaluation is taking longer than expected. Your responses may still be saved. Please contact the administrator if you do not receive confirmation.'
            : '⚠️ There was an issue saving your responses. Please contact the administrator.';
          setMessages((prev) => [
            ...prev,
            {
              id: 'save-error',
              type: 'system',
              content: errorMessage,
              timestamp: new Date(),
            },
          ]);
        }
      } else if (data.confirmation_question && !data.next_question) {
        // Only confirmation question returned - user needs to respond to it first
        setIsAwaitingConfirmation(true);
        setMessages((prev) => [
          ...prev,
          {
            id: `q-${Date.now()}`,
            type: 'system',
            content: data.confirmation_question,
            timestamp: new Date(),
          },
        ]);
      } else if (data.next_question) {
        // Regular next question (confirmation already handled or no confirmation)
        setIsAwaitingConfirmation(false);
        setMessages((prev) => [
          ...prev,
          {
            id: `q-${Date.now()}`,
            type: 'system',
            content: data.next_question,
            timestamp: new Date(),
          },
        ]);
      }
    } catch (err) {
      console.error(err);
      setMessages((prev) => [
        ...prev,
        {
          id: `error-${Date.now()}`,
          type: 'system',
          content: '⚠️ Error submitting response. Please try again.',
          timestamp: new Date(),
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const getPhaseLabel = (phase: string) => {
    switch (phase) {
      case 'predefined':
        return 'Part 1: Getting to Know You';
      case 'main':
        return 'Part 2: Follow-up Questions';
      case 'repeat':
        return 'Part 3: Clarification';
      case 'complete':
        return 'Complete';
      default:
        return phase;
    }
  };

  if (!sessionData) {
    return (
      <div className="flex min-h-screen items-center justify-center">
        <div className="animate-spin h-8 w-8 border-4 border-blue-600 border-t-transparent rounded-full"></div>
      </div>
    );
  }

  return (
    <main className="flex flex-col h-screen bg-gray-100">
      {/* Header */}
      <header className="bg-white shadow-sm px-4 py-3 flex items-center justify-between">
        <div>
          <h1 className="text-lg font-semibold text-gray-800">Human Interview</h1>
          <p className="text-sm text-gray-500">Participant: {sessionData.name}</p>
        </div>
        {progress && (
          <div className="text-right">
            <p className="text-sm font-medium text-blue-600">{getPhaseLabel(progress.phase)}</p>
            <p className="text-xs text-gray-500">
              Question {progress.current + 1} of ~{progress.total}
            </p>
          </div>
        )}
      </header>

      {/* Progress bar */}
      {progress && (
        <div className="bg-gray-200 h-1">
          <div
            className="bg-blue-600 h-1 transition-all duration-300"
            style={{ width: `${(progress.current / progress.total) * 100}%` }}
          />
        </div>
      )}

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.map((msg) => (
          <div
            key={msg.id}
            className={`flex ${msg.type === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            <div
              className={`max-w-[80%] rounded-lg px-4 py-2 ${
                msg.type === 'user'
                  ? 'bg-blue-600 text-white'
                  : msg.type === 'instruction'
                  ? 'bg-yellow-50 border border-yellow-200 text-gray-700'
                  : 'bg-white shadow text-gray-800'
              }`}
            >
              {msg.type === 'instruction' && (
                <p className="text-xs font-semibold text-yellow-700 mb-2">📋 Instructions</p>
              )}
              <p className="whitespace-pre-wrap text-sm">{msg.content}</p>
            </div>
          </div>
        ))}

        {isLoading && (
          <div className="flex justify-start">
            <div className="bg-white shadow rounded-lg px-4 py-2">
              <div className="flex space-x-1">
                <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
                <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
                <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
              </div>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      {!isComplete && (
        <form onSubmit={handleSubmit} className="bg-white border-t px-4 py-3">
          <div className="flex space-x-2">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Type your response..."
              disabled={isLoading}
              className="flex-1 px-4 py-2 border border-gray-300 rounded-full focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none transition disabled:bg-gray-100"
            />
            <button
              type="submit"
              disabled={isLoading || !input.trim()}
              className="bg-blue-600 text-white px-6 py-2 rounded-full hover:bg-blue-700 transition disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Send
            </button>
          </div>
          <p className="text-xs text-gray-400 mt-2 text-center">
            Press Enter to send • You may decline to answer any question
          </p>
        </form>
      )}

      {isComplete && (
        <div className="bg-green-50 border-t border-green-200 px-4 py-4 text-center">
          <p className="text-green-700 font-medium">Interview Complete! 🎉</p>
          <p className="text-sm text-green-600 mt-1">
            Thank you for your participation. Evaluation may take 1-2 minutes. Please do not close this page until you see a confirmation message.
          </p>
        </div>
      )}
    </main>
  );
}
