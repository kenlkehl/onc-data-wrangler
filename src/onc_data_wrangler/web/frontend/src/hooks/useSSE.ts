import { useCallback, useRef, useState } from 'react'
import { createSSEStream } from '@/api/client'
import type { ChatMessage } from '@/api/types'

let msgIdCounter = 0
function nextMsgId() {
  return `msg-${++msgIdCounter}-${Date.now()}`
}

interface UseSSEChatOptions {
  url: string
  sessionIdKey?: string
}

export function useSSEChat({ url }: UseSSEChatOptions) {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [isStreaming, setIsStreaming] = useState(false)
  const [sessionId, setSessionId] = useState<string>('')
  const [pendingAskUser, setPendingAskUser] = useState<{ toolUseId: string; question: string } | null>(null)
  const controllerRef = useRef<AbortController | null>(null)
  const accumulatedTextRef = useRef('')

  const sendMessage = useCallback(
    (text: string, extraBody?: Record<string, unknown>) => {
      setMessages((prev) => [
        ...prev,
        { id: nextMsgId(), role: 'user', content: text, timestamp: new Date() },
      ])
      setIsStreaming(true)
      accumulatedTextRef.current = ''

      const body = { message: text, session_id: sessionId, ...extraBody }

      controllerRef.current = createSSEStream(
        url,
        body,
        (event, data) => {
          try {
            const parsed = JSON.parse(data)
            switch (event) {
              case 'session':
                setSessionId(parsed.session_id)
                break
              case 'text':
                accumulatedTextRef.current += parsed.text
                setMessages((prev) => {
                  const last = prev[prev.length - 1]
                  if (last?.role === 'assistant') {
                    return [
                      ...prev.slice(0, -1),
                      { ...last, content: accumulatedTextRef.current },
                    ]
                  }
                  return [
                    ...prev,
                    {
                      id: nextMsgId(),
                      role: 'assistant',
                      content: accumulatedTextRef.current,
                      timestamp: new Date(),
                    },
                  ]
                })
                break
              case 'tool_call':
                setMessages((prev) => [
                  ...prev,
                  {
                    id: nextMsgId(),
                    role: 'tool',
                    content: `Calling ${parsed.tool}...`,
                    toolName: parsed.tool,
                    toolInput: parsed.input,
                    timestamp: new Date(),
                  },
                ])
                break
              case 'tool_result':
                setMessages((prev) => [
                  ...prev,
                  {
                    id: nextMsgId(),
                    role: 'tool',
                    content: parsed.result,
                    toolName: parsed.tool,
                    timestamp: new Date(),
                  },
                ])
                break
              case 'ask_user':
                setPendingAskUser({
                  toolUseId: parsed.tool_use_id,
                  question: parsed.question,
                })
                setMessages((prev) => [
                  ...prev,
                  {
                    id: nextMsgId(),
                    role: 'ask_user',
                    content: parsed.question,
                    timestamp: new Date(),
                  },
                ])
                break
              case 'error':
                setMessages((prev) => [
                  ...prev,
                  {
                    id: nextMsgId(),
                    role: 'error',
                    content: parsed.error,
                    timestamp: new Date(),
                  },
                ])
                break
              case 'done':
                accumulatedTextRef.current = ''
                break
            }
          } catch {
            // skip unparseable events
          }
        },
        () => {
          setIsStreaming(false)
        },
        (err) => {
          setIsStreaming(false)
          setMessages((prev) => [
            ...prev,
            {
              id: nextMsgId(),
              role: 'error',
              content: err.message,
              timestamp: new Date(),
            },
          ])
        }
      )
    },
    [url, sessionId]
  )

  const answerAskUser = useCallback(
    (answer: string) => {
      setPendingAskUser(null)
      setMessages((prev) => [
        ...prev,
        { id: nextMsgId(), role: 'user', content: answer, timestamp: new Date() },
      ])
      setIsStreaming(true)
      accumulatedTextRef.current = ''

      controllerRef.current = createSSEStream(
        url.replace('/chat', '/answer').replace('/message', '/message'),
        { answer, session_id: sessionId },
        (event, data) => {
          try {
            const parsed = JSON.parse(data)
            switch (event) {
              case 'text':
                accumulatedTextRef.current += parsed.text
                setMessages((prev) => {
                  const last = prev[prev.length - 1]
                  if (last?.role === 'assistant') {
                    return [
                      ...prev.slice(0, -1),
                      { ...last, content: accumulatedTextRef.current },
                    ]
                  }
                  return [
                    ...prev,
                    {
                      id: nextMsgId(),
                      role: 'assistant',
                      content: accumulatedTextRef.current,
                      timestamp: new Date(),
                    },
                  ]
                })
                break
              case 'tool_call':
                setMessages((prev) => [
                  ...prev,
                  {
                    id: nextMsgId(),
                    role: 'tool',
                    content: `Calling ${parsed.tool}...`,
                    toolName: parsed.tool,
                    toolInput: parsed.input,
                    timestamp: new Date(),
                  },
                ])
                break
              case 'tool_result':
                setMessages((prev) => [
                  ...prev,
                  {
                    id: nextMsgId(),
                    role: 'tool',
                    content: parsed.result,
                    toolName: parsed.tool,
                    timestamp: new Date(),
                  },
                ])
                break
              case 'ask_user':
                setPendingAskUser({
                  toolUseId: parsed.tool_use_id,
                  question: parsed.question,
                })
                setMessages((prev) => [
                  ...prev,
                  {
                    id: nextMsgId(),
                    role: 'ask_user',
                    content: parsed.question,
                    timestamp: new Date(),
                  },
                ])
                break
              case 'done':
                accumulatedTextRef.current = ''
                break
            }
          } catch {
            // skip
          }
        },
        () => setIsStreaming(false),
        (err) => {
          setIsStreaming(false)
          setMessages((prev) => [
            ...prev,
            { id: nextMsgId(), role: 'error', content: err.message, timestamp: new Date() },
          ])
        }
      )
    },
    [url, sessionId]
  )

  const reset = useCallback(() => {
    controllerRef.current?.abort()
    setMessages([])
    setSessionId('')
    setPendingAskUser(null)
    setIsStreaming(false)
    accumulatedTextRef.current = ''
  }, [])

  return {
    messages,
    isStreaming,
    sessionId,
    pendingAskUser,
    sendMessage,
    answerAskUser,
    reset,
  }
}
