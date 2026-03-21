import { useState, useEffect, useCallback } from 'react'
import { configApi } from '@/api/client'
import { useAppStore } from '@/stores/appStore'
import {
  Save,
  CheckCircle2,
  AlertCircle,
  Loader2,
  Plus,
  Trash2,
  FolderOpen,
} from 'lucide-react'
import type { ProjectConfig, OntologyInfo, FieldMapping } from '@/api/types'
import { BrowseDialog } from '@/components/data/BrowseDialog'

const TABS = [
  { id: 'project', label: 'Project' },
  { id: 'cohort', label: 'Cohort' },
  { id: 'extraction', label: 'Extraction' },
  { id: 'database', label: 'Database' },
  { id: 'query', label: 'Query' },
  { id: 'chatbot', label: 'Chatbot' },
  { id: 'field_mappings', label: 'Field Mappings' },
]

function TextInput({
  label,
  value,
  onChange,
  placeholder,
  type = 'text',
}: {
  label: string
  value: string | number | null | undefined
  onChange: (val: string) => void
  placeholder?: string
  type?: string
}) {
  return (
    <div>
      <label className="block text-sm font-medium text-gray-700 mb-1">{label}</label>
      <input
        type={type}
        value={value ?? ''}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        className="w-full border rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-primary-500 focus:border-transparent"
      />
    </div>
  )
}

function CheckboxInput({
  label,
  checked,
  onChange,
}: {
  label: string
  checked: boolean
  onChange: (val: boolean) => void
}) {
  return (
    <label className="flex items-center gap-2 text-sm">
      <input
        type="checkbox"
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
        className="rounded"
      />
      {label}
    </label>
  )
}

function ListInput({
  label,
  values,
  onChange,
  placeholder,
}: {
  label: string
  values: string[]
  onChange: (vals: string[]) => void
  placeholder?: string
}) {
  return (
    <div>
      <label className="block text-sm font-medium text-gray-700 mb-1">{label}</label>
      {values.map((val, i) => (
        <div key={i} className="flex gap-2 mb-1.5">
          <input
            type="text"
            value={val}
            onChange={(e) => {
              const newVals = [...values]
              newVals[i] = e.target.value
              onChange(newVals)
            }}
            placeholder={placeholder}
            className="flex-1 border rounded-lg px-3 py-1.5 text-sm focus:ring-2 focus:ring-primary-500 focus:border-transparent"
          />
          <button
            onClick={() => onChange(values.filter((_, j) => j !== i))}
            className="text-gray-400 hover:text-red-500"
          >
            <Trash2 size={14} />
          </button>
        </div>
      ))}
      <button
        onClick={() => onChange([...values, ''])}
        className="flex items-center gap-1 text-xs text-primary-600 hover:text-primary-700 mt-1"
      >
        <Plus size={12} />
        Add
      </button>
    </div>
  )
}

function BrowsableTextInput({
  label,
  value,
  onChange,
  placeholder,
  dirOnly = false,
  browseTitle,
}: {
  label: string
  value: string | null | undefined
  onChange: (val: string) => void
  placeholder?: string
  dirOnly?: boolean
  browseTitle?: string
}) {
  const [showBrowse, setShowBrowse] = useState(false)
  return (
    <div>
      <label className="block text-sm font-medium text-gray-700 mb-1">{label}</label>
      <div className="flex gap-2">
        <input
          type="text"
          value={value ?? ''}
          onChange={(e) => onChange(e.target.value)}
          placeholder={placeholder}
          className="flex-1 border rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-primary-500 focus:border-transparent"
        />
        <button
          type="button"
          onClick={() => setShowBrowse(true)}
          className="px-2.5 py-2 border rounded-lg text-gray-500 hover:bg-gray-50"
          title="Browse"
        >
          <FolderOpen size={14} />
        </button>
      </div>
      <BrowseDialog
        open={showBrowse}
        onClose={() => setShowBrowse(false)}
        onSelect={(paths) => onChange(paths[0])}
        dirOnly={dirOnly}
        title={browseTitle || label}
        initialPath={value || undefined}
      />
    </div>
  )
}

function BrowsableListInput({
  label,
  values,
  onChange,
  placeholder,
  dirOnly = false,
  browseTitle,
}: {
  label: string
  values: string[]
  onChange: (vals: string[]) => void
  placeholder?: string
  dirOnly?: boolean
  browseTitle?: string
}) {
  const [showBrowse, setShowBrowse] = useState(false)
  return (
    <div>
      <label className="block text-sm font-medium text-gray-700 mb-1">{label}</label>
      {values.map((val, i) => (
        <div key={i} className="flex gap-2 mb-1.5">
          <input
            type="text"
            value={val}
            onChange={(e) => {
              const newVals = [...values]
              newVals[i] = e.target.value
              onChange(newVals)
            }}
            placeholder={placeholder}
            className="flex-1 border rounded-lg px-3 py-1.5 text-sm focus:ring-2 focus:ring-primary-500 focus:border-transparent"
          />
          <button
            onClick={() => onChange(values.filter((_, j) => j !== i))}
            className="text-gray-400 hover:text-red-500"
          >
            <Trash2 size={14} />
          </button>
        </div>
      ))}
      <div className="flex gap-2 mt-1">
        <button
          onClick={() => onChange([...values, ''])}
          className="flex items-center gap-1 text-xs text-primary-600 hover:text-primary-700"
        >
          <Plus size={12} />
          Add
        </button>
        <button
          type="button"
          onClick={() => setShowBrowse(true)}
          className="flex items-center gap-1 text-xs text-gray-500 hover:text-gray-700"
        >
          <FolderOpen size={12} />
          Browse
        </button>
      </div>
      <BrowseDialog
        open={showBrowse}
        onClose={() => setShowBrowse(false)}
        onSelect={(paths) => {
          const filtered = paths.filter((p) => !values.includes(p))
          onChange([...values.filter(Boolean), ...filtered])
        }}
        multiple
        dirOnly={dirOnly}
        title={browseTitle || label}
      />
    </div>
  )
}

export function ConfigPage() {
  const configPath = useAppStore((s) => s.configPath)
  const setConfigPath = useAppStore((s) => s.setConfigPath)
  const [pathInput, setPathInput] = useState(configPath || '')
  const [config, setConfig] = useState<ProjectConfig | null>(null)
  const [activeTab, setActiveTab] = useState('project')
  const [isLoading, setIsLoading] = useState(false)
  const [isSaving, setIsSaving] = useState(false)
  const [saveStatus, setSaveStatus] = useState<'idle' | 'success' | 'error'>('idle')
  const [errors, setErrors] = useState<string[]>([])
  const [ontologies, setOntologies] = useState<OntologyInfo[]>([])

  useEffect(() => {
    configApi.ontologies().then(setOntologies).catch(() => {})
  }, [])

  const handleLoad = useCallback(async () => {
    if (!pathInput.trim()) return
    setIsLoading(true)
    try {
      const data = await configApi.load(pathInput.trim())
      setConfig(data as unknown as ProjectConfig)
      setConfigPath(pathInput.trim())
      setErrors([])
    } catch (err) {
      setErrors([(err as Error).message])
    } finally {
      setIsLoading(false)
    }
  }, [pathInput, setConfigPath])

  const handleSave = useCallback(async () => {
    if (!config || !pathInput.trim()) return
    setIsSaving(true)
    setSaveStatus('idle')
    try {
      await configApi.save(pathInput.trim(), config as unknown as Record<string, unknown>)
      setSaveStatus('success')
      setTimeout(() => setSaveStatus('idle'), 3000)
    } catch (err) {
      setSaveStatus('error')
      setErrors([(err as Error).message])
    } finally {
      setIsSaving(false)
    }
  }, [config, pathInput])

  const updateConfig = useCallback(
    (path: string, value: unknown) => {
      if (!config) return
      const newConfig = JSON.parse(JSON.stringify(config))
      const parts = path.split('.')
      let obj: Record<string, unknown> = newConfig
      for (let i = 0; i < parts.length - 1; i++) {
        obj = obj[parts[i]] as Record<string, unknown>
      }
      obj[parts[parts.length - 1]] = value
      setConfig(newConfig)
    },
    [config]
  )

  if (!config) {
    return (
      <div className="flex-1 flex items-center justify-center p-8">
        <div className="max-w-md w-full">
          <h2 className="text-2xl font-bold mb-6">Config Editor</h2>
          <div className="flex gap-2">
            <input
              type="text"
              value={pathInput}
              onChange={(e) => setPathInput(e.target.value)}
              placeholder="Path to config YAML..."
              className="flex-1 border rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-primary-500 focus:border-transparent"
              onKeyDown={(e) => e.key === 'Enter' && handleLoad()}
            />
            <button
              onClick={handleLoad}
              disabled={isLoading}
              className="flex items-center gap-2 px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 disabled:bg-gray-300 transition-colors text-sm"
            >
              {isLoading ? <Loader2 size={16} className="animate-spin" /> : <FolderOpen size={16} />}
              Load
            </button>
          </div>
          {errors.length > 0 && (
            <div className="mt-3 p-3 bg-red-50 border border-red-200 rounded-lg text-sm text-red-700">
              {errors.map((e, i) => (
                <div key={i}>{e}</div>
              ))}
            </div>
          )}
        </div>
      </div>
    )
  }

  return (
    <div className="flex-1 flex flex-col h-full overflow-hidden">
      {/* Header with save */}
      <div className="flex items-center justify-between px-4 py-3 border-b bg-white">
        <div className="flex items-center gap-2">
          <span className="text-sm font-medium text-gray-600">{pathInput}</span>
          {saveStatus === 'success' && (
            <span className="flex items-center gap-1 text-xs text-green-600">
              <CheckCircle2 size={12} /> Saved
            </span>
          )}
          {saveStatus === 'error' && (
            <span className="flex items-center gap-1 text-xs text-red-600">
              <AlertCircle size={12} /> Error saving
            </span>
          )}
        </div>
        <button
          onClick={handleSave}
          disabled={isSaving}
          className="flex items-center gap-2 px-3 py-1.5 bg-primary-600 text-white rounded-lg hover:bg-primary-700 disabled:bg-gray-300 transition-colors text-sm"
        >
          {isSaving ? <Loader2 size={14} className="animate-spin" /> : <Save size={14} />}
          Save
        </button>
      </div>

      {/* Tabs */}
      <div className="flex border-b bg-gray-50 px-4">
        {TABS.map(({ id, label }) => (
          <button
            key={id}
            onClick={() => setActiveTab(id)}
            className={`px-4 py-2.5 text-sm font-medium border-b-2 transition-colors ${
              activeTab === id
                ? 'border-primary-500 text-primary-700'
                : 'border-transparent text-gray-500 hover:text-gray-700'
            }`}
          >
            {label}
          </button>
        ))}
      </div>

      {/* Tab content */}
      <div className="flex-1 overflow-y-auto p-6">
        <div className="max-w-2xl space-y-5">
          {activeTab === 'project' && (
            <>
              <TextInput
                label="Project Name"
                value={config.project.name}
                onChange={(v) => updateConfig('project.name', v)}
              />
              <BrowsableListInput
                label="Input Paths"
                values={config.project.input_paths || []}
                onChange={(v) => updateConfig('project.input_paths', v)}
                placeholder="/path/to/data"
                browseTitle="Select Input Data Sources"
              />
              <BrowsableTextInput
                label="Output Directory"
                value={config.project.output_dir}
                onChange={(v) => updateConfig('project.output_dir', v)}
                dirOnly
                browseTitle="Select Output Directory"
              />
            </>
          )}

          {activeTab === 'cohort' && config.cohort && (
            <>
              <BrowsableTextInput
                label="Patient File"
                value={config.cohort.patient_file}
                onChange={(v) => updateConfig('cohort.patient_file', v)}
                browseTitle="Select Patient File"
              />
              <TextInput
                label="Patient ID Column"
                value={config.cohort.patient_id_column}
                onChange={(v) => updateConfig('cohort.patient_id_column', v)}
              />
              <BrowsableTextInput
                label="Diagnosis File"
                value={config.cohort.diagnosis_file}
                onChange={(v) => updateConfig('cohort.diagnosis_file', v)}
                browseTitle="Select Diagnosis File"
              />
              <TextInput
                label="Diagnosis Code Column"
                value={config.cohort.diagnosis_code_column}
                onChange={(v) => updateConfig('cohort.diagnosis_code_column', v)}
              />
              <ListInput
                label="Diagnosis Code Filter"
                values={config.cohort.diagnosis_code_filter || []}
                onChange={(v) => updateConfig('cohort.diagnosis_code_filter', v)}
                placeholder="C34"
              />
              <TextInput label="Sex Column" value={config.cohort.sex_column} onChange={(v) => updateConfig('cohort.sex_column', v)} />
              <TextInput label="Race Column" value={config.cohort.race_column} onChange={(v) => updateConfig('cohort.race_column', v)} />
              <TextInput label="Birth Date Column" value={config.cohort.birth_date_column} onChange={(v) => updateConfig('cohort.birth_date_column', v)} />
              <TextInput label="Death Date Column" value={config.cohort.death_date_column} onChange={(v) => updateConfig('cohort.death_date_column', v)} />
              <TextInput label="Follow-up Date" value={config.cohort.followup_date} onChange={(v) => updateConfig('cohort.followup_date', v)} placeholder="YYYY-MM-DD" />
            </>
          )}

          {activeTab === 'extraction' && config.extraction && (
            <>
              <div className="grid grid-cols-2 gap-4">
                <TextInput label="LLM Provider" value={config.extraction.llm.provider} onChange={(v) => updateConfig('extraction.llm.provider', v)} />
                <TextInput label="Model" value={config.extraction.llm.model} onChange={(v) => updateConfig('extraction.llm.model', v)} />
              </div>
              <TextInput label="Base URL" value={config.extraction.llm.base_url} onChange={(v) => updateConfig('extraction.llm.base_url', v)} />
              <div className="grid grid-cols-2 gap-4">
                <TextInput label="Max Tokens" value={config.extraction.llm.max_tokens} onChange={(v) => updateConfig('extraction.llm.max_tokens', parseInt(v) || 0)} type="number" />
                <TextInput label="Temperature" value={config.extraction.llm.temperature} onChange={(v) => updateConfig('extraction.llm.temperature', parseFloat(v) || 0)} type="number" />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Ontologies</label>
                <div className="space-y-1.5">
                  {ontologies.map((ont) => (
                    <label key={ont.id} className="flex items-start gap-2 text-sm">
                      <input
                        type="checkbox"
                        checked={(config.extraction.ontology_ids || []).includes(ont.id)}
                        onChange={(e) => {
                          const ids = config.extraction.ontology_ids || []
                          updateConfig(
                            'extraction.ontology_ids',
                            e.target.checked ? [...ids, ont.id] : ids.filter((i: string) => i !== ont.id)
                          )
                        }}
                        className="rounded mt-0.5"
                      />
                      <div>
                        <span className="font-medium">{ont.display_name}</span>
                        <span className="text-gray-500 ml-1">({ont.id})</span>
                        <p className="text-xs text-gray-400">{ont.description}</p>
                      </div>
                    </label>
                  ))}
                </div>
              </div>
              <TextInput label="Cancer Type" value={config.extraction.cancer_type} onChange={(v) => updateConfig('extraction.cancer_type', v)} />
              <BrowsableListInput label="Notes Paths" values={config.extraction.notes_paths || []} onChange={(v) => updateConfig('extraction.notes_paths', v)} placeholder="/path/to/notes" browseTitle="Select Notes Files" />
              <div className="grid grid-cols-2 gap-4">
                <TextInput label="Chunk Tokens" value={config.extraction.chunk_tokens} onChange={(v) => updateConfig('extraction.chunk_tokens', parseInt(v) || 0)} type="number" />
                <TextInput label="Patient Workers" value={config.extraction.patient_workers} onChange={(v) => updateConfig('extraction.patient_workers', parseInt(v) || 0)} type="number" />
              </div>
            </>
          )}

          {activeTab === 'database' && config.database && (
            <>
              <TextInput label="Record ID Prefix" value={config.database.record_id_prefix} onChange={(v) => updateConfig('database.record_id_prefix', v)} />
              <TextInput label="Min Non-Missing" value={config.database.min_non_missing} onChange={(v) => updateConfig('database.min_non_missing', parseInt(v) || 0)} type="number" />
              <CheckboxInput label="De-identify Dates" checked={config.database.deidentify_dates} onChange={(v) => updateConfig('database.deidentify_dates', v)} />
              <ListInput label="Forbidden Output Columns" values={config.database.forbidden_output_columns || []} onChange={(v) => updateConfig('database.forbidden_output_columns', v)} />
            </>
          )}

          {activeTab === 'query' && config.query && (
            <>
              <div className="grid grid-cols-2 gap-4">
                <TextInput label="Min Cell Size" value={config.query.min_cell_size} onChange={(v) => updateConfig('query.min_cell_size', parseInt(v) || 0)} type="number" />
                <TextInput label="Max Query Rows" value={config.query.max_query_rows} onChange={(v) => updateConfig('query.max_query_rows', parseInt(v) || 0)} type="number" />
              </div>
              <TextInput label="MCP Host" value={config.query.mcp_host} onChange={(v) => updateConfig('query.mcp_host', v)} />
              <TextInput label="MCP Port" value={config.query.mcp_port} onChange={(v) => updateConfig('query.mcp_port', parseInt(v) || 0)} type="number" />
            </>
          )}

          {activeTab === 'chatbot' && config.chatbot && (
            <>
              <TextInput label="Chatbot Name" value={config.chatbot.chatbot_name} onChange={(v) => updateConfig('chatbot.chatbot_name', v)} />
              <div className="grid grid-cols-2 gap-4">
                <TextInput label="LLM Provider" value={config.chatbot.llm.provider} onChange={(v) => updateConfig('chatbot.llm.provider', v)} />
                <TextInput label="Model" value={config.chatbot.llm.model} onChange={(v) => updateConfig('chatbot.llm.model', v)} />
              </div>
              <TextInput label="MCP URL" value={config.chatbot.mcp_url} onChange={(v) => updateConfig('chatbot.mcp_url', v)} />
              <div className="grid grid-cols-2 gap-4">
                <TextInput label="Host" value={config.chatbot.host} onChange={(v) => updateConfig('chatbot.host', v)} />
                <TextInput label="Port" value={config.chatbot.port} onChange={(v) => updateConfig('chatbot.port', parseInt(v) || 0)} type="number" />
              </div>
            </>
          )}

          {activeTab === 'field_mappings' && (
            <FieldMappingsEditor
              mappings={config.field_mappings || {}}
              onChange={(v) => updateConfig('field_mappings', v)}
            />
          )}
        </div>
      </div>
    </div>
  )
}

function FieldMappingsEditor({
  mappings,
  onChange,
}: {
  mappings: Record<string, FieldMapping[]>
  onChange: (v: Record<string, FieldMapping[]>) => void
}) {
  const categories = Object.keys(mappings)
  const [newCategory, setNewCategory] = useState('')

  const addCategory = () => {
    if (!newCategory.trim()) return
    onChange({ ...mappings, [newCategory.trim()]: [] })
    setNewCategory('')
  }

  const addMapping = (category: string) => {
    onChange({
      ...mappings,
      [category]: [...(mappings[category] || []), { source: '', target: '' }],
    })
  }

  const updateMapping = (category: string, index: number, field: keyof FieldMapping, value: string) => {
    const updated = [...(mappings[category] || [])]
    updated[index] = { ...updated[index], [field]: value }
    onChange({ ...mappings, [category]: updated })
  }

  const removeMapping = (category: string, index: number) => {
    onChange({
      ...mappings,
      [category]: mappings[category].filter((_, i) => i !== index),
    })
  }

  const removeCategory = (category: string) => {
    const updated = { ...mappings }
    delete updated[category]
    onChange(updated)
  }

  return (
    <div className="space-y-6">
      {categories.map((cat) => (
        <div key={cat} className="border rounded-lg p-4">
          <div className="flex items-center justify-between mb-3">
            <h4 className="font-medium text-sm">{cat}</h4>
            <button onClick={() => removeCategory(cat)} className="text-xs text-red-500 hover:text-red-700">
              Remove category
            </button>
          </div>
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-xs text-gray-500">
                <th className="pb-1">Source Column</th>
                <th className="pb-1">Target Field</th>
                <th className="pb-1">Transform</th>
                <th className="pb-1 w-8"></th>
              </tr>
            </thead>
            <tbody>
              {(mappings[cat] || []).map((mapping, i) => (
                <tr key={i}>
                  <td className="pr-2 pb-1.5">
                    <input
                      type="text"
                      value={mapping.source}
                      onChange={(e) => updateMapping(cat, i, 'source', e.target.value)}
                      className="w-full border rounded px-2 py-1 text-xs"
                    />
                  </td>
                  <td className="pr-2 pb-1.5">
                    <input
                      type="text"
                      value={mapping.target}
                      onChange={(e) => updateMapping(cat, i, 'target', e.target.value)}
                      className="w-full border rounded px-2 py-1 text-xs"
                    />
                  </td>
                  <td className="pr-2 pb-1.5">
                    <input
                      type="text"
                      value={mapping.transform || ''}
                      onChange={(e) => updateMapping(cat, i, 'transform', e.target.value)}
                      className="w-full border rounded px-2 py-1 text-xs"
                      placeholder="none"
                    />
                  </td>
                  <td className="pb-1.5">
                    <button onClick={() => removeMapping(cat, i)} className="text-gray-400 hover:text-red-500">
                      <Trash2 size={12} />
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          <button
            onClick={() => addMapping(cat)}
            className="flex items-center gap-1 text-xs text-primary-600 hover:text-primary-700 mt-2"
          >
            <Plus size={12} /> Add mapping
          </button>
        </div>
      ))}

      <div className="flex gap-2">
        <input
          type="text"
          value={newCategory}
          onChange={(e) => setNewCategory(e.target.value)}
          placeholder="New category name..."
          className="border rounded-lg px-3 py-1.5 text-sm flex-1"
          onKeyDown={(e) => e.key === 'Enter' && addCategory()}
        />
        <button
          onClick={addCategory}
          className="flex items-center gap-1 px-3 py-1.5 bg-gray-100 hover:bg-gray-200 rounded-lg text-sm"
        >
          <Plus size={14} /> Add Category
        </button>
      </div>
    </div>
  )
}
