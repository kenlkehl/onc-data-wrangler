import { useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { ChevronDown, ChevronRight, Wrench, AlertCircle, HelpCircle } from 'lucide-react'
import type { ChatMessage } from '@/api/types'

interface MessageBubbleProps {
  message: ChatMessage
  showToolCalls?: boolean
}

export function MessageBubble({ message, showToolCalls = false }: MessageBubbleProps) {
  const [expanded, setExpanded] = useState(false)

  if (message.role === 'user') {
    return (
      <div className="flex justify-end mb-3">
        <div className="max-w-[75%] bg-primary-50 text-primary-900 rounded-2xl rounded-br-md px-4 py-2.5 text-sm">
          {message.content}
        </div>
      </div>
    )
  }

  if (message.role === 'assistant') {
    return (
      <div className="flex justify-start mb-3">
        <div className="max-w-[85%] bg-white border border-gray-200 rounded-2xl rounded-bl-md px-4 py-2.5 text-sm shadow-sm">
          <div className="markdown-content">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>
              {message.content}
            </ReactMarkdown>
          </div>
        </div>
      </div>
    )
  }

  if (message.role === 'tool') {
    if (!showToolCalls) return null
    return (
      <div className="flex justify-start mb-2">
        <div className="max-w-[85%] bg-amber-50 border border-amber-200 rounded-lg px-3 py-2 text-xs">
          <button
            onClick={() => setExpanded(!expanded)}
            className="flex items-center gap-1.5 text-amber-700 font-medium"
          >
            <Wrench size={12} />
            {message.toolName && <span>{message.toolName}</span>}
            {expanded ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
          </button>
          {expanded && (
            <pre className="mt-2 text-xs text-gray-600 whitespace-pre-wrap overflow-x-auto max-h-48 overflow-y-auto">
              {message.content}
            </pre>
          )}
        </div>
      </div>
    )
  }

  if (message.role === 'ask_user') {
    return (
      <div className="flex justify-start mb-3">
        <div className="max-w-[85%] bg-purple-50 border border-purple-200 rounded-2xl rounded-bl-md px-4 py-2.5 text-sm">
          <div className="flex items-start gap-2">
            <HelpCircle size={16} className="text-purple-500 mt-0.5 flex-shrink-0" />
            <div className="markdown-content">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                {message.content}
              </ReactMarkdown>
            </div>
          </div>
        </div>
      </div>
    )
  }

  if (message.role === 'error') {
    return (
      <div className="flex justify-center mb-3">
        <div className="bg-red-50 border border-red-200 rounded-lg px-4 py-2 text-sm text-red-700 flex items-center gap-2">
          <AlertCircle size={16} />
          {message.content}
        </div>
      </div>
    )
  }

  return null
}
