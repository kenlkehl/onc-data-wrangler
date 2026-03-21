import { useState, useEffect, useRef, useCallback } from 'react'
import { pipelineApi } from '@/api/client'
import { createSSEStream } from '@/api/client'
import { useAppStore } from '@/stores/appStore'
import {
  Play,
  Square,
  CheckCircle2,
  XCircle,
  Clock,
  Loader2,
  RotateCcw,
  FileText,
  FolderOpen,
} from 'lucide-react'
import { BrowseDialog } from '@/components/data/BrowseDialog'

const ALL_STAGES = [
  { id: 'cohort', label: 'Cohort' },
  { id: 'prepare_notes', label: 'Prepare Notes' },
  { id: 'extract', label: 'Extract' },
  { id: 'harmonize', label: 'Harmonize' },
  { id: 'propose_tables', label: 'Propose Tables' },
  { id: 'database', label: 'Database' },
  { id: 'metadata', label: 'Metadata' },
]

interface StageStatus {
  name: string
  status: 'pending' | 'running' | 'completed' | 'failed' | 'skipped'
  started_at?: string
  completed_at?: string
  progress?: { current: number; total: number; message: string }
}

interface RunStatus {
  run_id: string
  status: 'running' | 'completed' | 'failed'
  current_stage: string | null
  stages: StageStatus[]
  error?: string
}

function StageIcon({ status }: { status: string }) {
  switch (status) {
    case 'completed':
      return <CheckCircle2 size={18} className="text-green-500" />
    case 'running':
      return <Loader2 size={18} className="text-primary-500 animate-spin" />
    case 'failed':
      return <XCircle size={18} className="text-red-500" />
    case 'skipped':
      return <Clock size={18} className="text-gray-400" />
    default:
      return <Clock size={18} className="text-gray-300" />
  }
}

export function PipelinePage() {
  const configPath = useAppStore((s) => s.configPath)
  const [configInput, setConfigInput] = useState(configPath || '')
  const [selectedStages, setSelectedStages] = useState<string[]>([])
  const [resume, setResume] = useState(false)
  const [runId, setRunId] = useState<string | null>(null)
  const [runStatus, setRunStatus] = useState<RunStatus | null>(null)
  const [logs, setLogs] = useState<{ timestamp: string; level: string; message: string; stage: string }[]>([])
  const [isStarting, setIsStarting] = useState(false)
  const [showBrowse, setShowBrowse] = useState(false)
  const logRef = useRef<HTMLDivElement>(null)
  const pollingRef = useRef<ReturnType<typeof setInterval> | null>(null)

  // Poll status while running
  useEffect(() => {
    if (!runId) return
    const poll = async () => {
      try {
        const status = (await pipelineApi.status(runId)) as unknown as RunStatus
        setRunStatus(status)
        if (status.status !== 'running') {
          if (pollingRef.current) clearInterval(pollingRef.current)
        }
      } catch {
        // ignore
      }
    }
    poll()
    pollingRef.current = setInterval(poll, 2000)
    return () => {
      if (pollingRef.current) clearInterval(pollingRef.current)
    }
  }, [runId])

  // Subscribe to log stream
  useEffect(() => {
    if (!runId) return
    const controller = createSSEStream(
      `/api/pipeline/${runId}/logs`,
      {},
      (event, data) => {
        if (event === 'log') {
          try {
            const entry = JSON.parse(data)
            setLogs((prev) => [...prev, entry])
          } catch {
            // ignore
          }
        }
      },
      () => {},
      () => {}
    )
    return () => controller.abort()
  }, [runId])

  // Auto-scroll logs
  useEffect(() => {
    if (logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight
    }
  }, [logs])

  const handleRun = useCallback(async () => {
    if (!configInput.trim()) return
    setIsStarting(true)
    setLogs([])
    try {
      const stages = selectedStages.length > 0 ? selectedStages : undefined
      const res = await pipelineApi.run(configInput.trim(), stages, resume)
      setRunId(res.run_id)
    } catch (err) {
      console.error('Failed to start pipeline:', err)
    } finally {
      setIsStarting(false)
    }
  }, [configInput, selectedStages, resume])

  const toggleStage = (stageId: string) => {
    setSelectedStages((prev) =>
      prev.includes(stageId) ? prev.filter((s) => s !== stageId) : [...prev, stageId]
    )
  }

  const isRunning = runStatus?.status === 'running'

  return (
    <div className="flex-1 flex flex-col h-full overflow-hidden">
      {/* Controls */}
      <div className="p-4 border-b bg-white">
        <div className="flex items-center gap-3">
          <input
            type="text"
            value={configInput}
            onChange={(e) => setConfigInput(e.target.value)}
            placeholder="Path to config YAML..."
            className="flex-1 border rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-primary-500 focus:border-transparent"
          />
          <button
            onClick={() => setShowBrowse(true)}
            className="flex items-center gap-1.5 px-3 py-2 border rounded-lg text-sm text-gray-600 hover:bg-gray-50"
          >
            <FolderOpen size={14} />
          </button>
          <BrowseDialog
            open={showBrowse}
            onClose={() => setShowBrowse(false)}
            onSelect={(paths) => setConfigInput(paths[0])}
            title="Select Config YAML"
          />
          <label className="flex items-center gap-1.5 text-sm text-gray-600">
            <input
              type="checkbox"
              checked={resume}
              onChange={(e) => setResume(e.target.checked)}
              className="rounded"
            />
            <RotateCcw size={14} />
            Resume
          </label>
          <button
            onClick={handleRun}
            disabled={isRunning || isStarting || !configInput.trim()}
            className="flex items-center gap-2 px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 disabled:bg-gray-300 transition-colors text-sm"
          >
            {isStarting ? (
              <Loader2 size={16} className="animate-spin" />
            ) : (
              <Play size={16} />
            )}
            Run Pipeline
          </button>
        </div>
        {/* Stage selection */}
        <div className="flex flex-wrap gap-2 mt-3">
          {ALL_STAGES.map(({ id, label }) => (
            <button
              key={id}
              onClick={() => toggleStage(id)}
              className={`px-2.5 py-1 text-xs rounded-full border transition-colors ${
                selectedStages.includes(id)
                  ? 'bg-primary-100 border-primary-300 text-primary-700'
                  : 'bg-white border-gray-200 text-gray-500 hover:border-gray-300'
              }`}
            >
              {label}
            </button>
          ))}
          {selectedStages.length > 0 && (
            <button
              onClick={() => setSelectedStages([])}
              className="text-xs text-gray-400 hover:text-gray-600 underline"
            >
              Clear (run all)
            </button>
          )}
        </div>
      </div>

      {/* Main content */}
      <div className="flex-1 flex overflow-hidden">
        {/* Stage list */}
        <div className="w-72 border-r overflow-y-auto bg-gray-50 p-3 space-y-2">
          {(runStatus?.stages || ALL_STAGES.map((s) => ({ name: s.id, status: 'pending' as const }))).map(
            (stage) => {
              const label = ALL_STAGES.find((s) => s.id === stage.name)?.label || stage.name
              const stageData = 'progress' in stage ? (stage as StageStatus) : null
              return (
                <div
                  key={stage.name}
                  className={`p-3 rounded-lg border bg-white ${
                    stage.status === 'running' ? 'border-primary-300 ring-1 ring-primary-100' : 'border-gray-200'
                  }`}
                >
                  <div className="flex items-center gap-2">
                    <StageIcon status={stage.status} />
                    <span className="text-sm font-medium">{label}</span>
                  </div>
                  {stageData?.progress && stageData.progress.total > 0 && (
                    <div className="mt-2">
                      <div className="flex justify-between text-xs text-gray-500 mb-1">
                        <span>{stageData.progress.message}</span>
                        <span>
                          {stageData.progress.current}/{stageData.progress.total}
                        </span>
                      </div>
                      <div className="w-full bg-gray-200 rounded-full h-1.5">
                        <div
                          className="bg-primary-500 h-1.5 rounded-full transition-all"
                          style={{
                            width: `${(stageData.progress.current / stageData.progress.total) * 100}%`,
                          }}
                        />
                      </div>
                    </div>
                  )}
                </div>
              )
            }
          )}
          {runStatus?.error && (
            <div className="p-3 rounded-lg bg-red-50 border border-red-200 text-sm text-red-700">
              {runStatus.error}
            </div>
          )}
        </div>

        {/* Log viewer */}
        <div className="flex-1 flex flex-col overflow-hidden">
          <div className="flex items-center gap-2 px-4 py-2 border-b bg-gray-50">
            <FileText size={14} className="text-gray-500" />
            <span className="text-xs font-medium text-gray-600">Logs</span>
            <span className="text-xs text-gray-400">({logs.length} entries)</span>
          </div>
          <div ref={logRef} className="flex-1 overflow-y-auto bg-gray-900 p-3 font-mono text-xs">
            {logs.length === 0 && (
              <p className="text-gray-500">Pipeline logs will appear here...</p>
            )}
            {logs.map((entry, i) => (
              <div key={i} className="leading-5">
                <span className="text-gray-500">{new Date(entry.timestamp).toLocaleTimeString()} </span>
                <span
                  className={
                    entry.level === 'ERROR'
                      ? 'text-red-400'
                      : entry.level === 'WARNING'
                      ? 'text-yellow-400'
                      : 'text-gray-300'
                  }
                >
                  [{entry.level}]
                </span>
                {entry.stage && <span className="text-blue-400"> [{entry.stage}]</span>}
                <span className="text-gray-200"> {entry.message}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}
