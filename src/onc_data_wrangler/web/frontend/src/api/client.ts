// API client for Talk-to-Data backend

const BASE_URL = ''

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`, {
    headers: { 'Content-Type': 'application/json', ...options?.headers },
    ...options,
  })
  if (!res.ok) {
    const text = await res.text()
    throw new Error(`API error ${res.status}: ${text}`)
  }
  return res.json()
}

// Config API
export const configApi = {
  load: (path: string) =>
    request<Record<string, unknown>>(`/api/config/load?path=${encodeURIComponent(path)}`),

  save: (path: string, config: Record<string, unknown>) =>
    request<{ status: string }>('/api/config/save', {
      method: 'PUT',
      body: JSON.stringify({ path, config }),
    }),

  validate: (config: Record<string, unknown>) =>
    request<{ valid: boolean; errors: string[] }>('/api/config/validate', {
      method: 'POST',
      body: JSON.stringify({ config }),
    }),

  ontologies: () => request<{ id: string; display_name: string; description: string; version: string }[]>('/api/config/ontologies'),

  ontologyFields: (id: string) =>
    request<{ categories: { id: string; name: string; items: { id: string; name: string; data_type: string; description: string }[] }[] }>(
      `/api/config/ontology/${id}/fields`
    ),
}

// Data API
export const dataApi = {
  files: (paths: string[]) =>
    request<{ path: string; name: string; size_bytes: number; type: string; columns: { name: string; type: string }[]; row_count: number | null }[]>(
      `/api/data/files?paths=${encodeURIComponent(paths.join(','))}`
    ),

  preview: (path: string, limit = 50, offset = 0) =>
    request<{ columns: string[]; rows: unknown[][]; total_rows: number }>(
      `/api/data/preview?path=${encodeURIComponent(path)}&limit=${limit}&offset=${offset}`
    ),

  columnStats: (path: string, column: string) =>
    request<Record<string, unknown>>(
      `/api/data/column-stats?path=${encodeURIComponent(path)}&column=${encodeURIComponent(column)}`
    ),

  outputs: (outputDir: string) =>
    request<{ files: { name: string; path: string; size_bytes: number; type: string }[] }>(
      `/api/data/outputs?output_dir=${encodeURIComponent(outputDir)}`
    ),

  browse: (path?: string) =>
    request<{
      current_path: string
      parent: string | null
      entries: { name: string; path: string; is_dir: boolean; size_bytes?: number; ext?: string }[]
    }>(`/api/data/browse${path ? `?path=${encodeURIComponent(path)}` : ''}`),

  mkdir: (path: string) =>
    request<{ path: string }>('/api/data/mkdir', {
      method: 'POST',
      body: JSON.stringify({ path }),
    }),
}

// Pipeline API
export const pipelineApi = {
  run: (configPath: string, stages?: string[], resume = false) =>
    request<{ run_id: string }>('/api/pipeline/run', {
      method: 'POST',
      body: JSON.stringify({ config_path: configPath, stages, resume }),
    }),

  status: (runId: string) =>
    request<Record<string, unknown>>(`/api/pipeline/${runId}/status`),

  runs: () => request<Record<string, unknown>[]>('/api/pipeline/runs'),
}

// Setup API
export const setupApi = {
  start: (params: {
    data_paths?: string[];
    output_dir?: string;
    config_path?: string;
    provider?: string;
    model?: string;
    ollama_base_url?: string;
  }) =>
    request<{ session_id: string }>('/api/setup/start', {
      method: 'POST',
      body: JSON.stringify(params),
    }),

  config: (sessionId: string) =>
    request<{ yaml: string; parsed: Record<string, unknown> | null }>(
      `/api/setup/${sessionId}/config`
    ),

  stop: (sessionId: string) =>
    request<{ status: string }>(`/api/setup/${sessionId}`, { method: 'DELETE' }),

  ollamaModels: (baseUrl = 'http://localhost:11434') =>
    request<{
      available: boolean;
      models: { name: string; size: number; modified_at: string }[];
      error?: string;
    }>(`/api/setup/ollama/models?base_url=${encodeURIComponent(baseUrl)}`),
}

// SSE helpers

export function createSSEStream(
  url: string,
  body: Record<string, unknown>,
  onEvent: (event: string, data: string) => void,
  onDone: () => void,
  onError: (err: Error) => void
): AbortController {
  const controller = new AbortController()

  fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    signal: controller.signal,
  })
    .then(async (res) => {
      if (!res.ok) {
        throw new Error(`SSE error ${res.status}`)
      }
      const reader = res.body?.getReader()
      if (!reader) throw new Error('No reader')

      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() || ''

        let currentEvent = ''
        let currentData = ''

        for (const line of lines) {
          if (line.startsWith('event: ')) {
            currentEvent = line.slice(7)
          } else if (line.startsWith('data: ')) {
            currentData += (currentData ? '\n' : '') + line.slice(6)
          } else if (line === '' && currentEvent) {
            onEvent(currentEvent, currentData)
            if (currentEvent === 'done') {
              onDone()
            }
            currentEvent = ''
            currentData = ''
          }
        }
      }
      onDone()
    })
    .catch((err) => {
      if (err.name !== 'AbortError') {
        onError(err)
      }
    })

  return controller
}
