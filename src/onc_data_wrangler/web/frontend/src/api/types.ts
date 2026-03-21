// TypeScript interfaces matching Python dataclasses in config.py

export interface LLMConfig {
  provider: string
  model: string
  base_url: string
  api_key: string
  max_tokens: number
  temperature: number
  vertex_project: string
  vertex_region: string
}

export interface VLLMServerConfig {
  gpus: number[]
  gpus_per_server: number
  base_port: number
  extra_args: string[]
}

export interface CohortStageConfig {
  patient_file: string | null
  diagnosis_file: string | null
  demographics_file: string | null
  demographics_files: string[]
  patient_id_column: string
  diagnosis_code_column: string | null
  diagnosis_code_filter: string[]
  sex_column: string | null
  race_column: string | null
  ethnicity_column: string | null
  birth_date_column: string | null
  death_date_column: string | null
  death_indicator_column: string | null
  followup_date: string | null
  id_prefix: string
}

export interface ExtractionConfig {
  llm: LLMConfig
  vllm_servers: VLLMServerConfig
  ontology_ids: string[]
  cancer_type: string
  chunk_tokens: number
  overlap_tokens: number
  max_retries: number
  patient_workers: number
  patient_id_column: string
  notes_text_column: string
  notes_date_column: string
  notes_type_column: string
  notes_paths: string[]
}

export interface DatabaseConfig {
  record_id_prefix: string
  min_non_missing: number
  forbidden_output_columns: string[]
  deidentify_dates: boolean
}

export interface QueryConfig {
  min_cell_size: number
  max_query_rows: number
  max_output_fraction: number
  mcp_host: string
  mcp_port: number
}

export interface ChatbotConfig {
  llm: LLMConfig
  mcp_url: string
  mcp_token: string
  max_agent_turns: number
  host: string
  port: number
  chatbot_name: string
}

export interface FieldMapping {
  source: string
  target: string
  transform?: string
  value_map?: Record<string, string>
}

export interface ProjectConfig {
  project: {
    name: string
    input_paths: string[]
    output_dir: string
    max_budget_usd: number
  }
  cohort: CohortStageConfig
  extraction: ExtractionConfig
  database: DatabaseConfig
  query: QueryConfig
  chatbot: ChatbotConfig
  field_mappings: Record<string, FieldMapping[]>
  patient_id_columns: Record<string, string>
}

// API response types

export interface FileInfo {
  path: string
  name: string
  size_bytes: number
  type: 'csv' | 'parquet'
  columns: { name: string; type: string }[]
  row_count: number | null
}

export interface DataPreview {
  columns: string[]
  rows: (string | number | null)[][]
  total_rows: number
}

export interface ColumnStatsResult {
  column: string
  dtype: string
  non_null_count: number
  unique_count: number
  top_values?: { value: string; count: number }[]
  numeric_stats?: {
    min: number
    max: number
    mean: number
    median: number
  }
}

export interface OntologyInfo {
  id: string
  display_name: string
  description: string
  version: string
}

export interface OntologyField {
  id: string
  name: string
  data_type: string
  description: string
  valid_values?: string[]
}

export interface OntologyCategory {
  id: string
  name: string
  items: OntologyField[]
}

export interface StageStatus {
  name: string
  status: 'pending' | 'running' | 'completed' | 'failed' | 'skipped'
  started_at?: string
  completed_at?: string
  progress?: {
    current: number
    total: number
    message: string
  }
}

export interface PipelineRunStatus {
  run_id: string
  config_path: string
  status: 'running' | 'completed' | 'failed'
  current_stage: string | null
  stages: StageStatus[]
  error?: string
}

export interface LogEntry {
  timestamp: string
  level: string
  message: string
  stage: string
}

export interface SetupSession {
  session_id: string
}

// SSE event types (shared between chatbot and setup agent)
export interface SSETextEvent {
  text: string
}

export interface SSEToolCallEvent {
  tool: string
  input: Record<string, unknown>
}

export interface SSEToolResultEvent {
  tool: string
  result: string
}

export interface SSEAskUserEvent {
  tool_use_id: string
  question: string
}

// Chat message types for display
export interface ChatMessage {
  id: string
  role: 'user' | 'assistant' | 'tool' | 'error' | 'ask_user'
  content: string
  toolName?: string
  toolInput?: Record<string, unknown>
  timestamp: Date
}
