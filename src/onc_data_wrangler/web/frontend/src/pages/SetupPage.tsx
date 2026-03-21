import { useState, useEffect, useCallback } from 'react'
import { ChatPanel } from '@/components/chat/ChatPanel'
import { useSSEChat } from '@/hooks/useSSE'
import { setupApi } from '@/api/client'
import { useAppStore } from '@/stores/appStore'
import { Loader2, FileText, CheckCircle2, Circle, FolderOpen, AlertCircle } from 'lucide-react'
import { BrowseDialog } from '@/components/data/BrowseDialog'

const SETUP_STAGES = [
  'Project Basics',
  'Data Exploration',
  'Cohort Definition',
  'Notes Configuration',
  'Ontology Selection',
  'Field Mappings',
  'Proposed Tables',
  'Database & Query',
  'Summary & Next Steps',
]

function detectStage(messages: { content: string; role: string }[]): number {
  for (let i = messages.length - 1; i >= 0; i--) {
    const msg = messages[i]
    if (msg.role !== 'assistant') continue
    const text = msg.content.toLowerCase()
    for (let s = SETUP_STAGES.length - 1; s >= 0; s--) {
      if (text.includes(`stage ${s + 1}`) || text.includes(SETUP_STAGES[s].toLowerCase())) {
        return s
      }
    }
  }
  return 0
}

export function SetupPage() {
  const [sessionId, setSessionId] = useState<string | null>(null)
  const [isStarting, setIsStarting] = useState(false)
  const [dataPaths, setDataPaths] = useState('')
  const [outputDir, setOutputDir] = useState('')
  const [configYaml, setConfigYaml] = useState('')
  const setConfigPath = useAppStore((s) => s.setConfigPath)

  const [browseTarget, setBrowseTarget] = useState<'data' | 'output' | null>(null)

  // Provider selection state
  const [provider, setProvider] = useState<'claude' | 'ollama'>('claude')
  const [ollamaUrl, setOllamaUrl] = useState('http://localhost:11434')
  const [ollamaModel, setOllamaModel] = useState('')
  const [ollamaModels, setOllamaModels] = useState<{ name: string; size: number }[]>([])
  const [ollamaError, setOllamaError] = useState<string | null>(null)
  const [ollamaLoading, setOllamaLoading] = useState(false)

  // Fetch Ollama models when provider is ollama
  useEffect(() => {
    if (provider !== 'ollama') return
    let cancelled = false
    setOllamaLoading(true)
    setOllamaError(null)
    setupApi
      .ollamaModels(ollamaUrl)
      .then((res) => {
        if (cancelled) return
        if (res.available) {
          setOllamaModels(res.models)
          if (res.models.length > 0 && !ollamaModel) {
            setOllamaModel(res.models[0].name)
          }
          setOllamaError(null)
        } else {
          setOllamaModels([])
          setOllamaError(res.error || 'Ollama server not reachable')
        }
      })
      .catch((err) => {
        if (cancelled) return
        setOllamaModels([])
        setOllamaError(String(err))
      })
      .finally(() => {
        if (!cancelled) setOllamaLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [provider, ollamaUrl])

  const chat = useSSEChat({ url: '/api/setup/message' })
  const currentStage = detectStage(chat.messages)

  // Poll config preview while session is active
  useEffect(() => {
    if (!sessionId) return
    const interval = setInterval(async () => {
      try {
        const res = await setupApi.config(sessionId)
        setConfigYaml(res.yaml || '')
      } catch {
        // config not yet written
      }
    }, 5000)
    return () => clearInterval(interval)
  }, [sessionId])

  const handleStart = useCallback(async () => {
    setIsStarting(true)
    try {
      const paths = dataPaths.split('\n').map((p) => p.trim()).filter(Boolean)
      const res = await setupApi.start({
        data_paths: paths.length > 0 ? paths : undefined,
        output_dir: outputDir.trim() || undefined,
        provider,
        model: provider === 'ollama' ? ollamaModel : undefined,
        ollama_base_url: provider === 'ollama' ? ollamaUrl : undefined,
      })
      setSessionId(res.session_id)
      // Send initial message to trigger the agent
      // Pass session_id explicitly since React state update hasn't committed yet
      chat.sendMessage('Start the setup process.', { session_id: res.session_id })
    } catch (err) {
      console.error('Failed to start setup:', err)
    } finally {
      setIsStarting(false)
    }
  }, [dataPaths, outputDir, provider, ollamaModel, ollamaUrl, chat])

  if (!sessionId) {
    return (
      <div className="flex-1 flex items-center justify-center p-8">
        <div className="max-w-md w-full">
          <h2 className="text-2xl font-bold mb-6">New Project Setup</h2>
          <p className="text-gray-600 text-sm mb-6">
            Start an interactive session with the AI agent to configure your data wrangling project.
          </p>
          <div className="space-y-4">
            {/* Provider Selection */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Agent Provider
              </label>
              <div className="flex gap-3">
                <button
                  type="button"
                  onClick={() => setProvider('claude')}
                  className={`flex-1 px-3 py-2 border rounded-lg text-sm transition-colors ${
                    provider === 'claude'
                      ? 'border-primary-500 bg-primary-50 text-primary-700 font-medium'
                      : 'text-gray-600 hover:bg-gray-50'
                  }`}
                >
                  Claude (API)
                </button>
                <button
                  type="button"
                  onClick={() => setProvider('ollama')}
                  className={`flex-1 px-3 py-2 border rounded-lg text-sm transition-colors ${
                    provider === 'ollama'
                      ? 'border-primary-500 bg-primary-50 text-primary-700 font-medium'
                      : 'text-gray-600 hover:bg-gray-50'
                  }`}
                >
                  Ollama (Local)
                </button>
              </div>
            </div>

            {/* Ollama Settings (shown when Ollama selected) */}
            {provider === 'ollama' && (
              <div className="space-y-3 p-3 bg-amber-50 border border-amber-200 rounded-lg">
                <p className="text-xs text-amber-800">
                  Requires an Ollama server running locally. Install from{' '}
                  <span className="font-medium">ollama.com</span>, then pull a model:{' '}
                  <code className="bg-amber-100 px-1 rounded">ollama pull llama3.1:70b</code>
                </p>
                <div>
                  <label className="block text-xs font-medium text-gray-700 mb-1">
                    Ollama URL
                  </label>
                  <input
                    type="text"
                    value={ollamaUrl}
                    onChange={(e) => setOllamaUrl(e.target.value)}
                    placeholder="http://localhost:11434"
                    className="w-full border rounded-lg px-3 py-1.5 text-sm focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                  />
                </div>
                <div>
                  <label className="block text-xs font-medium text-gray-700 mb-1">
                    Model
                  </label>
                  {ollamaLoading ? (
                    <div className="flex items-center gap-2 text-sm text-gray-500 py-1.5">
                      <Loader2 size={14} className="animate-spin" />
                      Checking Ollama server...
                    </div>
                  ) : ollamaError ? (
                    <div className="space-y-2">
                      <div className="flex items-center gap-2 text-sm text-red-600">
                        <AlertCircle size={14} />
                        {ollamaError}
                      </div>
                      <input
                        type="text"
                        value={ollamaModel}
                        onChange={(e) => setOllamaModel(e.target.value)}
                        placeholder="Enter model name (e.g. llama3.1:70b)"
                        className="w-full border rounded-lg px-3 py-1.5 text-sm focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                      />
                    </div>
                  ) : ollamaModels.length > 0 ? (
                    <select
                      value={ollamaModel}
                      onChange={(e) => setOllamaModel(e.target.value)}
                      className="w-full border rounded-lg px-3 py-1.5 text-sm focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                    >
                      {ollamaModels.map((m) => (
                        <option key={m.name} value={m.name}>
                          {m.name} ({(m.size / 1024 ** 3).toFixed(1)} GB)
                        </option>
                      ))}
                    </select>
                  ) : (
                    <input
                      type="text"
                      value={ollamaModel}
                      onChange={(e) => setOllamaModel(e.target.value)}
                      placeholder="Enter model name (e.g. llama3.1:70b)"
                      className="w-full border rounded-lg px-3 py-1.5 text-sm focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                    />
                  )}
                </div>
              </div>
            )}

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Source Data Paths
              </label>
              <div className="flex gap-2">
                <textarea
                  value={dataPaths}
                  onChange={(e) => setDataPaths(e.target.value)}
                  placeholder="/path/to/data/dir&#10;/path/to/another/file.csv"
                  rows={3}
                  className="flex-1 border rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                />
                <button
                  type="button"
                  onClick={() => setBrowseTarget('data')}
                  className="self-start px-3 py-2 border rounded-lg text-sm text-gray-600 hover:bg-gray-50 flex items-center gap-1.5"
                >
                  <FolderOpen size={14} />
                  Browse
                </button>
              </div>
              <p className="text-xs text-gray-500 mt-1">One path per line (optional - agent will ask if omitted)</p>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Output Directory
              </label>
              <div className="flex gap-2">
                <input
                  type="text"
                  value={outputDir}
                  onChange={(e) => setOutputDir(e.target.value)}
                  placeholder="/path/to/output"
                  className="flex-1 border rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                />
                <button
                  type="button"
                  onClick={() => setBrowseTarget('output')}
                  className="px-3 py-2 border rounded-lg text-sm text-gray-600 hover:bg-gray-50 flex items-center gap-1.5"
                >
                  <FolderOpen size={14} />
                  Browse
                </button>
              </div>
            </div>
            <button
              onClick={handleStart}
              disabled={isStarting || (provider === 'ollama' && !ollamaModel)}
              className="w-full bg-primary-600 text-white py-2.5 rounded-lg hover:bg-primary-700 disabled:bg-gray-300 transition-colors flex items-center justify-center gap-2"
            >
              {isStarting ? (
                <>
                  <Loader2 size={16} className="animate-spin" />
                  Starting agent...
                </>
              ) : (
                'Start Setup Agent'
              )}
            </button>
          </div>
          <BrowseDialog
            open={browseTarget !== null}
            onClose={() => setBrowseTarget(null)}
            onSelect={(paths) => {
              if (browseTarget === 'data') {
                const existing = dataPaths.trim()
                const newPaths = paths.join('\n')
                setDataPaths(existing ? `${existing}\n${newPaths}` : newPaths)
              } else if (browseTarget === 'output') {
                setOutputDir(paths[0])
              }
            }}
            multiple={browseTarget === 'data'}
            dirOnly={browseTarget === 'output'}
            title={browseTarget === 'data' ? 'Select Data Sources' : 'Select Output Directory'}
          />
        </div>
      </div>
    )
  }

  return (
    <div className="flex-1 flex h-full overflow-hidden">
      {/* Chat panel - left 60% */}
      <div className="w-3/5 border-r flex flex-col">
        <ChatPanel
          messages={chat.messages}
          isStreaming={chat.isStreaming}
          pendingAskUser={chat.pendingAskUser}
          onSend={(text) => chat.sendMessage(text)}
          onAnswer={(text) => chat.answerAskUser(text)}
          placeholder="Chat with the setup agent..."
        />
      </div>

      {/* Context panels - right 40% */}
      <div className="w-2/5 flex flex-col overflow-hidden">
        {/* Stage progress */}
        <div className="p-4 border-b">
          <h3 className="text-sm font-semibold text-gray-700 mb-3">Setup Progress</h3>
          <div className="space-y-1.5">
            {SETUP_STAGES.map((stage, i) => (
              <div
                key={stage}
                className={`flex items-center gap-2 text-xs py-1 ${
                  i === currentStage
                    ? 'text-primary-700 font-semibold'
                    : i < currentStage
                    ? 'text-green-600'
                    : 'text-gray-400'
                }`}
              >
                {i < currentStage ? (
                  <CheckCircle2 size={14} className="text-green-500" />
                ) : i === currentStage ? (
                  <Circle size={14} className="text-primary-500 fill-primary-500" />
                ) : (
                  <Circle size={14} />
                )}
                <span>
                  {i + 1}. {stage}
                </span>
              </div>
            ))}
          </div>
        </div>

        {/* Config preview */}
        <div className="flex-1 overflow-hidden flex flex-col">
          <div className="flex items-center gap-2 px-4 py-2 border-b bg-gray-50">
            <FileText size={14} className="text-gray-500" />
            <span className="text-xs font-medium text-gray-600">Config Preview</span>
          </div>
          <div className="flex-1 overflow-y-auto p-3">
            {configYaml ? (
              <pre className="text-xs text-gray-700 whitespace-pre-wrap font-mono">
                {configYaml}
              </pre>
            ) : (
              <p className="text-xs text-gray-400 italic">
                Config will appear here as the agent writes it...
              </p>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
