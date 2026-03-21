import { useEffect, useRef, useState } from 'react'
import { MessageBubble } from './MessageBubble'
import { ChatInput } from './ChatInput'
import { Eye, EyeOff, Loader2 } from 'lucide-react'
import type { ChatMessage } from '@/api/types'

interface ChatPanelProps {
  messages: ChatMessage[]
  isStreaming: boolean
  pendingAskUser: { toolUseId: string; question: string } | null
  onSend: (text: string) => void
  onAnswer?: (text: string) => void
  placeholder?: string
  className?: string
}

export function ChatPanel({
  messages,
  isStreaming,
  pendingAskUser,
  onSend,
  onAnswer,
  placeholder,
  className = '',
}: ChatPanelProps) {
  const [showToolCalls, setShowToolCalls] = useState(false)
  const scrollRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [messages])

  const handleSend = (text: string) => {
    if (pendingAskUser && onAnswer) {
      onAnswer(text)
    } else {
      onSend(text)
    }
  }

  return (
    <div className={`flex flex-col h-full ${className}`}>
      <div className="flex items-center justify-between px-4 py-2 border-b bg-gray-50">
        <span className="text-xs text-gray-500 font-medium">
          {isStreaming && (
            <span className="flex items-center gap-1.5">
              <Loader2 size={12} className="animate-spin" />
              Agent is thinking...
            </span>
          )}
        </span>
        <button
          onClick={() => setShowToolCalls(!showToolCalls)}
          className="flex items-center gap-1 text-xs text-gray-500 hover:text-gray-700"
        >
          {showToolCalls ? <EyeOff size={12} /> : <Eye size={12} />}
          {showToolCalls ? 'Hide' : 'Show'} tool calls
        </button>
      </div>
      <div ref={scrollRef} className="flex-1 overflow-y-auto p-4">
        {messages.length === 0 && (
          <div className="text-center text-gray-400 text-sm mt-12">
            Send a message to get started.
          </div>
        )}
        {messages.map((msg) => (
          <MessageBubble key={msg.id} message={msg} showToolCalls={showToolCalls} />
        ))}
      </div>
      <ChatInput
        onSend={handleSend}
        disabled={isStreaming}
        placeholder={
          pendingAskUser
            ? 'Answer the question above...'
            : placeholder
        }
      />
    </div>
  )
}
