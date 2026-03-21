import { create } from 'zustand'

interface AppState {
  configPath: string | null
  setConfigPath: (path: string | null) => void
  pipelineRunId: string | null
  setPipelineRunId: (id: string | null) => void
  setupSessionId: string | null
  setSetupSessionId: (id: string | null) => void
}

export const useAppStore = create<AppState>((set) => ({
  configPath: null,
  setConfigPath: (path) => set({ configPath: path }),
  pipelineRunId: null,
  setPipelineRunId: (id) => set({ pipelineRunId: id }),
  setupSessionId: null,
  setSetupSessionId: (id) => set({ setupSessionId: id }),
}))
