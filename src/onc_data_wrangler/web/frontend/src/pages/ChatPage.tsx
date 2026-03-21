import { useEffect, useState } from 'react'
import { ChatPanel } from '@/components/chat/ChatPanel'
import { useSSEChat } from '@/hooks/useSSE'
import { RotateCcw } from 'lucide-react'

interface SummaryStats {
  project_name?: string
  total_patients?: number
  tables?: { name: string; row_count: number; columns: string[] }[]
}

export function ChatPage() {
  const [summaryStats, setSummaryStats] = useState<SummaryStats | null>(null)
  const [chatbotName, setChatbotName] = useState('Talk-to-Data')
  const chat = useSSEChat({ url: '/chat' })

  useEffect(() => {
    fetch('/summary-stats')
      .then((r) => r.json())
      .then((data) => {
        if (!data.error) setSummaryStats(data)
      })
      .catch(() => {})
    fetch('/config')
      .then((r) => r.json())
      .then((data) => {
        if (data.chatbot_name) setChatbotName(data.chatbot_name)
      })
      .catch(() => {})
  }, [])

  const handleReset = async () => {
    if (chat.sessionId) {
      await fetch('/reset', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: chat.sessionId }),
      })
    }
    chat.reset()
  }

  return (
    <div className="flex-1 flex flex-col h-full overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b bg-white">
        <h2 className="text-lg font-semibold">{chatbotName}</h2>
        <button
          onClick={handleReset}
          className="flex items-center gap-1.5 text-xs text-gray-500 hover:text-gray-700 px-2 py-1 rounded hover:bg-gray-100"
        >
          <RotateCcw size={14} />
          Reset
        </button>
      </div>

      <div className="flex-1 flex overflow-hidden">
        {/* Summary sidebar */}
        {summaryStats && (
          <div className="w-72 border-r overflow-y-auto bg-gray-50 p-4">
            <h3 className="text-sm font-semibold text-gray-700 mb-3">Database Summary</h3>
            {summaryStats.project_name && (
              <p className="text-xs text-gray-500 mb-3">
                {summaryStats.project_name}
              </p>
            )}
            {summaryStats.total_patients != null && (
              <div className="bg-white rounded-lg border p-3 mb-3">
                <div className="text-2xl font-bold text-primary-600">
                  {summaryStats.total_patients.toLocaleString()}
                </div>
                <div className="text-xs text-gray-500">Total patients</div>
              </div>
            )}
            {summaryStats.tables &&
              summaryStats.tables.map((table) => (
                <div key={table.name} className="mb-2 p-2 bg-white rounded border text-xs">
                  <div className="font-medium">{table.name}</div>
                  <div className="text-gray-400">
                    {table.row_count?.toLocaleString()} rows &middot;{' '}
                    {table.columns?.length} columns
                  </div>
                </div>
              ))}
          </div>
        )}

        {/* Chat */}
        <div className="flex-1">
          <ChatPanel
            messages={chat.messages}
            isStreaming={chat.isStreaming}
            pendingAskUser={chat.pendingAskUser}
            onSend={(text) => chat.sendMessage(text)}
            onAnswer={(text) => chat.answerAskUser(text)}
            placeholder="Ask a question about your data..."
          />
        </div>
      </div>
    </div>
  )
}
